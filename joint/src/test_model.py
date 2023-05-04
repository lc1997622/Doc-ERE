import torch.nn as nn
from transformers import AutoConfig, AutoModel, RobertaConfig, RobertaModel,RobertaForMaskedLM
import torch
from .utils import to_cuda

class EventEncoder(nn.Module):
    def __init__(self, vocab_size, model_name="roberta-base", aggr="max"):
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
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
        return output


class EventDecoder(nn.Module):
    def __init__(self, model_name="roberta-base"):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, RobertaConfig):
            self.model = RobertaModel.from_pretrained(model_name)
            self.model.config.is_decoder = True
        else:
            raise NotImplementedError
        
    
    def forward(self, input_embedding,attention_mask):
        outputs = self.model(inputs_embeds = input_embedding,attention_mask = attention_mask).last_hidden_state
        return outputs

class Model(nn.Module):
    def __init__(self, vocab_size, model_name="roberta-base", embed_dim=768, aggr="mean"):
        nn.Module.__init__(self)
        self.encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.decoder = EventDecoder(model_name)
        self.test_linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.test_soft = nn.LogSoftmax(dim=1)
        self.aggr = aggr

    def forward(self, inputs):

        ## 文档编码
        encocder_output = self.encoder(inputs)

        ## 将文档的多个句子中的event提取出来，将event排列为一个序列
        event_spans = inputs["event_spans"]
        doc_splits = inputs["splits"]
        event_embed = []
        seq_embed = []
        atten_mask = []

        for i in range(0, len(doc_splits)-1):
            embed = []
            sembed = []
            doc_embed = encocder_output[doc_splits[i]:doc_splits[i+1]]
            doc_spans = event_spans[i]
            for j, spans in enumerate(doc_spans):
                sembed.append(doc_embed[j][0])
                for span in spans:
                    if self.aggr == "max" or self.aggr == "mean":
                        embed.extend(doc_embed[j][span[0]:span[1]])
                    else:
                        raise NotImplementedError
                    
            ## sentences embedding + events embedding. 
            # lenght of sembed  = num of sentences + num of events.
            sembed.extend(embed)

            pad_len = 512 - len(sembed)
            mask = [1] * len(sembed) + [0] * pad_len

            ### torch.stack(sembed)  shape : (len of sembed, 768)

            pad_tokens = to_cuda(torch.LongTensor([[1] * pad_len]))

            pad_tensor = self.decoder.model.embeddings.word_embeddings(pad_tokens)
            pad_tensor = torch.squeeze(pad_tensor)
            seq_embed_pad = torch.cat((torch.stack(sembed), pad_tensor),0)
            atten_mask.append(mask)
            seq_embed.append(seq_embed_pad)
        atten_mask = torch.LongTensor(atten_mask)

        decoder_output = self.decoder(input_embedding = torch.stack(seq_embed),attention_mask = to_cuda(atten_mask))
        outputs = self.test_linear(decoder_output)
        outputs = self.dropout(outputs)
        return outputs
    