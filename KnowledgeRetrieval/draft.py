from LLM.prompts import ANALYZE_MEDICAL_QUESTION_PROMPT,EXTRACT_MEDICAL_ENTITIES_FROM_TEXT_PROMPT
from Unit import *
from LLM.LLMBatch import *

############### Draft
def LLMForDraft(loadPath,savePath):
    qbank_Data = load_json(loadPath)
    save_file_kgs = []
    for index, item in enumerate(qbank_Data):
        question = item["question"]
        options = item["options"]
        message = question + "\n" + str(options)

        query = f"""{ANALYZE_MEDICAL_QUESTION_PROMPT}"""
        query = query.replace("{message}", message)

        item_index = {"custom_id": f'{index + 1}',
                      "method": "POST",
                      "url": "/v1/chat/completions",
                      "body": {"model": "gpt-5-mini-2025-08-07",
                               "messages": [
                                   {"role": "user", "content": query}
                               ]
                               }
                      }
        save_file_kgs.append(item_index)
    save_to_jsonl(save_file_kgs,savePath)

def extractAnswerDraft(responsePath,loadPath,saveFile):
    response_Data = load_to_jsonl(responsePath)
    qbankDatas = load_json(loadPath)

    save_datas = []
    for index, item in enumerate(response_Data):
        out_answer = item["response"]["body"]["choices"][0]["message"]["content"]
        answerJson = changeStrToJson(out_answer)
        qbankData = qbankDatas[index]
        if answerJson:
            qbankData["answer_llM_analyse"] = answerJson.get("Analysis")
            qbankData["chart5MAnswer"] = answerJson.get("Answer")



        save_datas.append(qbankData)

    save_json(save_datas,saveFile)


############### Ner
def LLMForDraftNer(loadPath,savePath):
    qbank_Data = load_json(loadPath)
    save_file_kgs = []
    num = 0
    sum = 0
    for index, item in enumerate(qbank_Data):
        answer_llM_analyse = item.get("answer_llM_analyse")
        answer_llM_analyse_A = ""
        answer_llM_analyse_B = ""
        answer_llM_analyse_C = ""
        answer_llM_analyse_D = ""
        answer_llM_analyse_E = ""
        if isinstance(answer_llM_analyse, dict):
            if "A" in answer_llM_analyse:
                answer_llM_analyse_A = answer_llM_analyse["A"]
                sum = sum + 1
            if "B" in answer_llM_analyse:
                answer_llM_analyse_B = answer_llM_analyse["B"]
                sum = sum + 1
            if "C" in answer_llM_analyse:
                answer_llM_analyse_C = answer_llM_analyse["C"]
                sum = sum + 1
            if "D" in answer_llM_analyse:
                answer_llM_analyse_D = answer_llM_analyse["D"]
                sum = sum + 1
            if "E" in answer_llM_analyse:
                answer_llM_analyse_E = answer_llM_analyse["E"]
                sum = sum + 1
        else:
            print(item["question"])

        if answer_llM_analyse_A and answer_llM_analyse_A != "":
            message = EXTRACT_MEDICAL_ENTITIES_FROM_TEXT_PROMPT.format(answer_llM_analyse=answer_llM_analyse_A)
            item_index = {"custom_id": f'{num + 1}',
                          "method": "POST",
                          "url": "/v1/chat/completions",
                          "body": {"model": "gpt-5-mini-2025-08-07",
                                   "messages": [
                                       {"role": "user", "content": message}
                                   ]
                                   }
                          }
            save_file_kgs.append(item_index)
            num = num + 1
        if answer_llM_analyse_B and answer_llM_analyse_B != "":
            message = EXTRACT_MEDICAL_ENTITIES_FROM_TEXT_PROMPT.format(answer_llM_analyse=answer_llM_analyse_B)
            item_index = {"custom_id": f'{num + 1}',
                          "method": "POST",
                          "url": "/v1/chat/completions",
                          "body": {"model": "gpt-5-mini-2025-08-07",
                                   "messages": [
                                       {"role": "user", "content": message}
                                   ]
                                   }
                          }
            save_file_kgs.append(item_index)
            num = num + 1
        if answer_llM_analyse_C and answer_llM_analyse_C != "":
            message = EXTRACT_MEDICAL_ENTITIES_FROM_TEXT_PROMPT.format(answer_llM_analyse=answer_llM_analyse_C)
            item_index = {"custom_id": f'{num + 1}',
                          "method": "POST",
                          "url": "/v1/chat/completions",
                          "body": {"model": "gpt-5-mini-2025-08-07",
                                   "messages": [
                                       {"role": "user", "content": message}
                                   ]
                                   }
                          }
            save_file_kgs.append(item_index)
            num = num + 1
        if answer_llM_analyse_D and answer_llM_analyse_D != "":
            message = EXTRACT_MEDICAL_ENTITIES_FROM_TEXT_PROMPT.format(answer_llM_analyse=answer_llM_analyse_D)
            item_index = {"custom_id": f'{num + 1}',
                          "method": "POST",
                          "url": "/v1/chat/completions",
                          "body": {"model": "gpt-5-mini-2025-08-07",
                                   "messages": [
                                       {"role": "user", "content": message}
                                   ]
                                   }
                          }
            save_file_kgs.append(item_index)
            num = num + 1
        if answer_llM_analyse_E and answer_llM_analyse_E != "":
           message = EXTRACT_MEDICAL_ENTITIES_FROM_TEXT_PROMPT.format(answer_llM_analyse=answer_llM_analyse_E)
           item_index = {"custom_id": f'{num + 1}',
                          "method": "POST",
                          "url": "/v1/chat/completions",
                          "body": {"model": "gpt-5-mini-2025-08-07",
                                   "messages": [
                                       {"role": "user", "content": message}
                                   ]
                                   }
                          }
           save_file_kgs.append(item_index)
           num = num + 1
    print(f"num = {num},sum = {sum}")
    save_to_jsonl(save_file_kgs,savePath)

def extractAnswerNer(responsePath,loadPath,savePath):
    response_Data = load_to_jsonl(responsePath)
    qbankDatas = load_json(loadPath)

    save_datas = []
    num = 0
    LLM_Analysis_Ner = {}
    tapi = 0
    i = 0
    for index, item in enumerate(response_Data):
        if tapi > index:
            continue
        out_answer = item["response"]["body"]["choices"][0]["message"]["content"]
        answerJson = changeStrToList(out_answer)
        if answerJson:
            qbankData = qbankDatas[num]
            options = qbankData.get("answer_llM_analyse")
            len_i = len(options)

            if i == 0:
                LLM_Analysis_Ner["A"] = answerJson
                i = i + 1
            elif i == 1:
                LLM_Analysis_Ner["B"] = answerJson
                i = i + 1
            elif i == 2:
                LLM_Analysis_Ner["C"] = answerJson
                i = i + 1
            elif i == 3:
                LLM_Analysis_Ner["D"] = answerJson
                i = i + 1
            elif i == 4:
                LLM_Analysis_Ner["E"] = answerJson
                i = i + 1

            if i == len_i:
                qbankData["LLM_Analysis_Ner"] = LLM_Analysis_Ner
                LLM_Analysis_Ner = {}
                num = num + 1
                i = 0
                save_datas.append(qbankData)
                tapi = index

        else:
            qbankData["LLM_Analysis_Ner"] = out_answer
            num = num + 1
            tapi = index + len_i
            save_datas.append(qbankData)

    save_json(save_datas,savePath)



if __name__ == '__main__':
    # # 1.draft
    qaData = "../QA/Tuberculosis.json"
    input_file = "../Tuberculosis/draft/PMID_All.jsonl"
    output_file = "../Tuberculosis/draft/PMID_All_output.jsonl"
    error_file = "../Tuberculosis/draft/PMID_All_error.jsonl"
    # 1.1.draft
    LLMForDraft(loadPath=qaData,savePath=input_file)
    # 1.2.draft batch
    batch_data(
        input_file_path=input_file,
        output_file_path=output_file,
        error_file_path=error_file,
    )

    # 2.ner
    draft_file = "../Tuberculosis/draft/draft.json"
    draftNer_file = "../Tuberculosis/draft/NER_All.jsonl"
    draftNer_out = "../Tuberculosis/draft/NER_out.jsonl"
    draftNer_error = "../Tuberculosis/draft/NER_error.jsonl"
    # extractAnswerDraft(responsePath=output_file,loadPath=qaData,saveFile=draft_file)
    # # 2.1.ner
    # LLMForDraftNer(draft_file,draftNer_file)
    # # 2.2.ner batch
    # batch_data(
    #     input_file_path=draftNer_file,
    #     output_file_path=draftNer_out,
    #     error_file_path=draftNer_error,
    # )
    draftNer_saveFile = "../Tuberculosis/draft/Tuberculosis.json"
    extractAnswerNer(draftNer_out,draft_file,draftNer_saveFile)