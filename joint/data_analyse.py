from src.data import myNoTensorDataset, get_dataloader


if __name__ == "__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    dataset =  myNoTensorDataset(tokenizer,"/home/lichao/ERE/MAVEN-ERE/data/MAVEN_ERE","train",128,False,None)

    eventMap = {}
    
    for ea in dataset.examples:
        events =  ea.events
        time_line = []
        cor = ea.coref_relations
        sub = ea.subevent_relations['subevent']
        bef = ea.temporal_relations['BEFORE']
        cas = ea.causal_relations['CAUSE']
        pre = ea.causal_relations['PRECONDITION']


        repeat_line = 0

        ### 处理时序关系

        # for re in bef:
        #     if re[0] not in time_line:
        #         time_line.append(re[0])
        #     if re[1] not in time_line:
        #         time_line.append(re[1])

        # for i in range(len(bef)):
        #     flag = 0
        #     for re in bef:
        #         if time_line.index(re[0]) > time_line.index(re[1]):
        #             ind1 = time_line.index(re[0])
        #             ind2 = time_line.index(re[1])
        #             time_line.pop(ind2)
        #             time_line.insert(ind1,re[1])
        #             flag = 2
                
        #         if bef.index(re) == len(bef) - 1 and flag == 0:
        #             flag = 1
        #     if flag == 1:
        #         break

        # for re in bef:
        #     if time_line.index(re[0]) > time_line.index(re[1]):
        #         print("eeeeeeeeeeeeeeeerror")      

        # copy_time_line = []
        # for i in range(len(time_line) - 1):
        #     copy_time_line.append(time_line[i])
        #     if [time_line[i],time_line[i+1]] not in bef:
        #         copy_time_line.append("----------------")


        print("length of repeat before relation:" + str(repeat_line))
        print("length of time line:" + str(len(time_line)))

        print('cor len :' + str(len(cor)))
        print('sub len :' + str(len(sub)))
        print('bef len :' + str(len(bef)))
        print('cas len :' + str(len(cas)))
        print('pre len :' + str(len(pre)))
        print('*****************total relation length :' + str(len(cor)+len(sub)+len(bef)+len(cas)+len(pre)))



"""         for event in events:
            if 'type_id' not in event.keys() :
                continue
            if event['type_id'] in eventMap.keys():
                eventMap[event['type_id']].append(event['type']) 
            else:
                eventMap[event['type_id']] = [event['type']] """

