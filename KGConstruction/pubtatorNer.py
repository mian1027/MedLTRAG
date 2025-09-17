from Unit import *
from LLM.prompts import EXTRACT_BIOMED_RELATIONS_PROMPT
from LLM.LLMBatch import batch_data

def read_literature():
    years = list(range(2005, 2026))
    abstracts = []
    year2literatures = {year: [] for year in years}
    for year in years:
        fileName = f"../Tuberculosis/byYear/PMID_{year}.pubtator"
        with open(fileName) as f:
            literature = {'entity': {}}
            for line in f.readlines():
                line = line.strip()
                if line == ''  and literature != {}:
                    for entity_id in literature['entity']:
                        literature['entity'][entity_id]['entity_name'] = list(literature['entity'][entity_id]['entity_name'])
                    if literature != {'entity': {}} and literature['abstract'] != abstracts:
                        year2literatures[year].append(literature)
                        abstracts.append(literature['abstract'])
                    literature = {'entity': {}}
                    continue
                if '|t|' in line:
                    literature['title'] = line.split('|t|')[1]
                elif '|a|' in line:
                    literature['abstract'] = line.split('|a|')[1]
                else:
                    line_list = line.split('\t')
                    if len(line_list) <= 4:
                        continue
                    if len(line_list) != 6:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], None
                    else:
                        entity_name, entity_type, entity_id = line_list[3], line_list[4], line_list[5]
                    if entity_id == '-':
                        continue
                    if entity_id not in literature['entity']:
                        literature['entity'][entity_id] = {'entity_name':set(), 'entity_type': entity_type}
                    literature['entity'][entity_id]['entity_name'].add(entity_name)

            entity_type = set()
    return year2literatures

def get_entity_name(entity_names):
    if len(entity_names) == 1:
        return entity_names[0]
    else:
        return '{} ({})'.format(entity_names[0], ', '.join(entity_names[1:]))

def main():
    extracted = []
    year2literatures = read_literature()
    num = 0
    for year, literatures in year2literatures.items():
        for literature in literatures:
            # time.sleep(1)
            title, abstract = literature['title'], literature['abstract']
            item = {
                'title': title,
                'abstract': abstract,
                'triplet':[]
            }
            entity_names = ', '.join([get_entity_name(entity_info['entity_name']) for entity_info in literature['entity'].values()])
            # final_prompt = prompts.BIOMED_QA_PROMPT.format(question=q)
            message = EXTRACT_BIOMED_RELATIONS_PROMPT.format(abstract=abstract, entities=entity_names)
            item = {"custom_id": f'{num + 1}',
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5-nano-2025-08-07",
                        "temperature": 1,
                        "messages": [
                            {"role": "user", "content": message}
                        ]
                    }
                    }
            num = num + 1
            extracted.append(item)
    return extracted




if __name__ == '__main__':
    # extracted = main()
    # batch_size = 1000
    # for i in range(0, len(extracted), batch_size):
    #     batch = extracted[i:i + batch_size]  # slice
    #     output_file = f"../Tuberculosis/byYear/PMID_intput_{i // batch_size}.jsonl"
    #     save_to_jsonl(batch, output_file)

    input_file_batch = f"../Tuberculosis/byYear/PMID_intput_5.jsonl"
    output_file = "../Tuberculosis/byYear/PMID_output_5.jsonl"
    error_file = "../Tuberculosis/byYear/PMID_error_5.jsonl"
    batch_data(
        input_file_path=input_file_batch,
        output_file_path=output_file,
        error_file_path=error_file,
    )
