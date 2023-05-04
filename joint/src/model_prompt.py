import torch.nn as nn
from transformers import AutoConfig, AutoModel, RobertaConfig, RobertaModel,RobertaForMaskedLM,AutoModelForMaskedLM,GPT2Model
import torch
from .utils import to_cuda
from torch.cuda.amp import autocast

# class EventEncoder(nn.Module):
#     def __init__(self, vocab_size, model_name="roberta-base", aggr="max"):
#         nn.Module.__init__(self)
#         config = AutoConfig.from_pretrained(model_name)
#         if isinstance(config, RobertaConfig):
#             self.model = RobertaModel.from_pretrained(model_name)
#         else:
#             raise NotImplementedError
#         self.model.resize_token_embeddings(vocab_size)
#         self.model = nn.DataParallel(self.model)
#         self.aggr = aggr
    
#     def forward(self, inputs):
#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
#         return output

class EventDecoder(nn.Module):
    def __init__(self, model_name="roberta-base"):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaModel.from_pretrained(model_name, add_pooling_layer = False)
            self.model.config.is_decoder = False
        else:
            raise NotImplementedError
        
    
    def forward(self, input_embedding,attention_mask):
        outputs = self.model(inputs_embeds = input_embedding,attention_mask = attention_mask)[0]
        return outputs

class Model(nn.Module):
    def __init__(self, vocab_size, model_name="/home/lichao/ERE/MAVEN-ERE/paramter/roberta", embed_dim=768, aggr="mean"):
        nn.Module.__init__(self)
        self.decoder = EventDecoder(model_name)
        self.test_linear = nn.Linear(embed_dim, 3)
        self.dropout = nn.Dropout(0.1)
        self.aggr = aggr


    @autocast() 
    def forward(self, inputs, doc_embeddings):

        doc_ids = inputs["doc_id"]
        doc_prompts = inputs['doc_prompt']
        events_prompts = inputs['events_prompt']
        first_eventpromts=inputs['first_eventprompt']
        second_eventprompts= inputs['second_eventprompt']
        causals =  inputs['causal_prompt']
        mask_indexs = inputs['mask_index']
        event_locations = inputs['event_locations']

        ## 文档编码
        seq_embeds = []
        atten_mask = []
        mask_token_indexs = []
        for i in range(len(doc_ids)):
            text_embeds = doc_embeddings[doc_ids[i]]['doc_embeds']
            events_embeds = doc_embeddings[doc_ids[i]]['event_embeds']
            causal = causals[i]
            
            ## context embedding
            doc_prompt_embed = torch.squeeze(self.decoder.model.embeddings.word_embeddings(to_cuda(doc_prompts[i])))
            events_prompt_embed = torch.squeeze(self.decoder.model.embeddings.word_embeddings(to_cuda(events_prompts[i])))
            
            doc_embed = torch.cat((doc_prompt_embed, text_embeds),0)
            event_embed = torch.cat((events_prompt_embed, events_embeds),0)

            context_embed = torch.cat((doc_embed, event_embed),0)


            ## relation embedding
            first_prompt_embed = torch.squeeze(self.decoder.model.embeddings.word_embeddings(to_cuda(first_eventpromts[i])))
            second_prompt_embed = torch.squeeze(self.decoder.model.embeddings.word_embeddings(to_cuda(second_eventprompts[i])))


            first_embed =  torch.cat((first_prompt_embed, events_embeds[event_locations[i][0]:event_locations[i][0]+1]),0)
            second_embed =  torch.cat((second_prompt_embed, events_embeds[event_locations[i][1]:event_locations[i][1]+1]),0)
            relation_embed =  torch.cat((first_embed, second_embed),0)

            pre_embed =  torch.cat((context_embed, relation_embed),0)
            
            pad_len = 256 - len(pre_embed) - len(causal)
            mask  = [1] * len(pre_embed) + [1] * len(causal)+ [0] * pad_len
            pad_tokens = to_cuda(torch.LongTensor([[1] * pad_len]))
            mask_token_index = mask_indexs[i] + len(pre_embed)
            mask_token_indexs.append(mask_token_index)


            causal_embed = self.decoder.model.embeddings.word_embeddings(to_cuda(causal))
            pad_tensor = self.decoder.model.embeddings.word_embeddings(pad_tokens)
            causal_embed = torch.squeeze(causal_embed)
            pad_tensor = torch.squeeze(pad_tensor)

            seq_embed = torch.cat((pre_embed, causal_embed),0)
            seq_embed_pad = torch.cat((seq_embed, pad_tensor),0)
            atten_mask.append(mask)
            seq_embeds.append(seq_embed_pad)

        atten_mask = torch.LongTensor(atten_mask)
        decoder_output = self.decoder(input_embedding = torch.stack(seq_embeds),attention_mask = to_cuda(atten_mask))
        outputs = []
        for i in range(0, len(doc_ids)):
            output = decoder_output[i,mask_token_indexs[i],:]
            outputs.append(output)
        outputs = torch.stack(outputs)
        outputs = self.test_linear(outputs)
        outputs = self.dropout(outputs)
        return outputs

class DocEventEncoder(nn.Module):
    def __init__(self, vocab_size, model_name="/home/lichao/ERE/MAVEN-ERE/paramter/roberta", aggr="mean"):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaModel.from_pretrained(model_name)
        else:
            raise NotImplementedError
        self.model.resize_token_embeddings(vocab_size)
        self.model = nn.DataParallel(self.model)
        self.aggr = aggr
    
    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        event_spans = inputs["event_spans"]
        doc_splits = inputs["splits"]
        doc_embeds = []
        event_embeds = []
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
        for i in range(0, len(doc_splits)-1):
            e_embed = []
            s_embed = []
            doc_embed = output[doc_splits[i]:doc_splits[i+1]]
            doc_spans = event_spans[i]
            for j, spans in enumerate(doc_spans):
                s_embed.append(doc_embed[j][0])
                for span in spans:
                    if self.aggr == "max":
                        e_embed.append(doc_embed[j][span[0]:span[1]].max(0)[0])
                    elif self.aggr == "mean":
                        e_embed.append(doc_embed[j][span[0]:span[1]].mean(0))
                    else:
                        raise NotImplementedError

            doc_embeds.append(torch.stack(s_embed))
            event_embeds.append(torch.stack(e_embed))
        return doc_embeds,event_embeds