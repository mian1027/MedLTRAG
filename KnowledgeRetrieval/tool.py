import pickle
import pandas as pd
from Unit import *
from igraph import Graph
import random
from tqdm import tqdm
import torch
from treeNode import *
from LLM.prompts import *
from LLM.LLM import *
from itertools import zip_longest

def match_the_entity(question_kg:[str]):
    with open(ENTITY_EMBEDDINGS_PATH, 'rb') as f1:
        entity_embeddings = pickle.load(f1)

    with open(KEYWORD_EMBEDDINGS_PATH, 'rb') as f2:
        keyword_embeddings = pickle.load(f2)

    match_kg = set()
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])

    for kg_entity in question_kg:

        try:
            keyword_index = keyword_embeddings["keywords"].index(kg_entity)
        except ValueError:
            continue

        kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

        batch_size = 10000
        for i in range(0, len(entity_embeddings_emb), batch_size):
            batch_emb = entity_embeddings_emb.iloc[i:i + batch_size]
            cos_similarities = cosine_similarity_manual(batch_emb, kg_entity_emb)[0]

            batch_match = np.where(cos_similarities >= THRESHOLD)[0]
            for j in batch_match:
                global_index = i + j
                if global_index < len(entity_embeddings["entities"]):
                    match_entity = entity_embeddings["entities"][global_index]
                    match_kg.add(match_entity)

    return list(match_kg)


def sort_by_letters(data):
    return sorted(data, key=lambda s: re.sub('[^a-zA-Z]', '', s).lower())

def abandon_rels(relation):
    if str(relation.lower()) in USELESS_RELATION_LIST:
        return True
    return False

def abandon_entity(entity):
    if str(entity).lower() in USELESS_ENTITY_LIST:
        return True
    return False

def relation_search_prune(tokenizer,model,query,entity_name, pre_relations, pre_head, G:Graph):
    try:
        idx = G.vs.find(name=entity_name).index
    except ValueError:
        return []
    tail_edges = G.es.select(_source=idx)
    tail_relations = list({e['relation'] for e in tail_edges if e['relation'] is not None})
    head_edges = G.es.select(_target=idx)
    head_relations = list({e['relation'] for e in head_edges if e['relation'] is not None})
    head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
    tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

    if pre_head == True:
        tail_relations = list(set(tail_relations) - {pre_relations})
    else:
        head_relations = list(set(head_relations) - {pre_relations})
    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))

    if len(head_relations) > MAX_RELATION:
        head_relations = middle_prune(tokenizer,model,query, head_relations, [], [entity_name] * len(head_relations), [True] * len(head_relations),max_kg=MAX_RELATION,isPruneRelations=True)
    if len(tail_relations) > MAX_RELATION:
        tail_relations= middle_prune(tokenizer,model,query, tail_relations, [],  [entity_name] * len(tail_relations), [False] * len(tail_relations),max_kg=MAX_RELATION,isPruneRelations=True)

    tail_relations = list(set(tail_relations))
    relations = []
    for relation in head_relations:
        relations.append({"entity": entity_name, "relation": relation, "head": True})
    for relation in tail_relations:
        relations.append({"entity": entity_name, "relation": relation, "head": False})
    return relations


def entity_search(tokenizer,model,query,entity, relation, G:Graph, head: bool):
    candidates = []
    try:
        idx = G.vs.find(name=entity).index
    except ValueError:
        return []

    if not head:
        out_edges = G.es.select(_source=idx)
        for e in out_edges:
            target_idx = e.target
            target_name = G.vs[target_idx]['name']
            if relation is None or e['relation'] == relation:
                candidates.append(target_name)
    else:
        in_edges = G.es.select(_target=idx)
        for e in in_edges:
            source_idx = e.source
            source_name = G.vs[source_idx]['name']
            if relation is None or e['relation'] == relation:
                candidates.append(source_name)
    candidates = [candidate for candidate in candidates if not abandon_entity(candidate)]
    if len(candidates) > MAX_ENTITY:
        candidates = middle_prune(tokenizer,model,query, [relation] * len(candidates),candidates,[entity] *len(candidates),[head] * len(candidates),MAX_ENTITY)
    return candidates


def update_history(entity_candidates, entity, total_candidates,  total_relations, total_topic_entities, total_head):
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)

    total_candidates.extend(entity_candidates)
    total_relations.extend(candidates_relation)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_relations, total_topic_entities, total_head

def middle_prune(tokenizer,model,query,total_relations, total_candidates, total_topic_entities, total_head,
                       max_kg = MAX_KG, isPruneRelations: bool = False):

    articles = []
    if isPruneRelations == True:
        for (i, relations) in enumerate(total_relations):
            candidates = ""
            if len(total_candidates) > i:
                candidates = total_candidates[i]

            head = False
            if len(total_head) > i:
                head = total_head[i]

            tops = ""
            if len(total_topic_entities) > i:
                tops = total_topic_entities[i]

            if head == True:
                node = candidates + " " + relations + " " + tops
                node = node.strip()
                articles.append(node)
            else:
                node = tops + " " + relations + " " + candidates
                node = node.strip()
                articles.append(node)
    else:
        for (i, head) in enumerate(total_head):
            candidates = ""
            if len(total_candidates) > i:
                candidates = total_candidates[i]

            relations = ""
            if len(total_relations) > i:
                relations = total_relations[i]

            tops = ""
            if len(total_topic_entities) > i:
                tops = total_topic_entities[i]

            if head == True:
                node = candidates + " " + relations + " " + tops
                node = node.strip()
                articles.append(node)
            else:
                node = tops + " " + relations + " " + candidates
                node = node.strip()
                articles.append(node)




    if len(articles) > max_kg:
        paired_data = list(zip_longest(articles, total_candidates, total_relations, fillvalue=None))
        paired_data = random.sample(paired_data, min(MAX_NUM, len(paired_data)))
        logits_groups = []
        candidates = []
        scored_candidates = []
        # batch
        batched_articles = [
            paired_data[i: i + BATCH_SIZE]
            for i in range(0, len(paired_data), BATCH_SIZE)
        ]
        for batch_data in tqdm(batched_articles,desc="Inner Progress", ncols=100, colour="blue", leave=True):
            pairs = [[query, article] for article, _ , _ in batch_data]

            try:
                mini_batch_size = 4
                for i in range(0, len(pairs), mini_batch_size):
                    batch_pairs = pairs[i:i + mini_batch_size]
                    batch_candidates = batch_data[i:i + mini_batch_size]

                    with torch.no_grad():
                        encoded = tokenizer(
                            batch_pairs,
                            truncation=True,
                            padding=True,
                            return_tensors="pt",
                            max_length=1024,
                        )

                        logits = model(**encoded).logits.squeeze(dim=1)
                        logits_groups.append(logits.cpu())

                    for logit, (article, cand, relations) in zip(logits, batch_candidates):
                        scored_candidates.append((logit.item(), cand if cand is not None else relations))

            except Exception as e:

                print(f"Error processing batch: {e}")
                continue

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_candidates[:max_kg]
        for score, cand in top_results:
            if score > 0:
                candidates.append(cand)

        return candidates

    else:
        if isPruneRelations == False:
            return total_candidates
        else:
            return total_relations

def paths_prune_agin(tokenizer,model,query,total_relations, total_candidates, total_topic_entities, total_head, depth):
    articles = []
    paths = []
    chains = []
    print("============= entity_prune_last %d." % len(total_head))
    for (i,head) in enumerate(total_head):
        candidates = ""
        if len(total_candidates) > i:
            candidates = total_candidates[i]

        relations = ""
        if len(total_relations) > i:
            relations = total_relations[i]

        tops = ""
        if len(total_topic_entities) > i:
            tops = total_topic_entities[i]


        if head == True:
            item = {"header": candidates, "relation": relations, "trail": tops, "head":True, "First":tops,"Last":candidates}
            node = str(item["header"]) + "->" + str(item["relation"]) + "->" + str(item["trail"])
            if node not in articles:
                articles.append(node)
                chains.append(item)
            else:
                print(f"Again ========== {node}")
        else:
            item = {"header": tops, "relation": relations, "trail": candidates,"head":False, "First":tops,"Last":candidates}
            node = str(item["header"]) + "->" + str(item["relation"]) + "->" + str(item["trail"])
            if node not in articles:
                articles.append(node)
                chains.append(item)
            else:
                print(f"Again ========== {node}")

    if len(articles) > MAX_KG:
        paired_data = list(zip(articles, chains))
        paired_data = random.sample(paired_data, min(MAX_NUM, len(paired_data)))
        logits_groups = []
        results = []
        # batch
        batch = BATCH_SIZE - (depth - 1) * 1
        batched_articles = [
            paired_data[i: i + batch]
            for i in range(0, len(paired_data), batch)
        ]

        for batch_data in tqdm(batched_articles,desc="Inner Progress", ncols=100, colour="blue", leave=True):
            pairs = [[query, article] for article, _ in batch_data]
            try:
                with torch.no_grad():
                    encoded = tokenizer(
                        pairs,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=2048,
                    )

                logits = model(**encoded).logits.squeeze(dim=1)

                for logit, (_, chain) in zip(logits.cpu(), batch_data):
                    logits_groups.append((logit.item(), chain))
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        logits_groups.sort(key=lambda x: x[0], reverse=True)
        top_results = logits_groups[:MAX_KG]
        for score, path in top_results:
            trail = path.get("Last")
            match_kg = match_the_entity([trail])
            if len(match_kg) > 0:
                path["match_kg"] = match_kg
            results.append(path)

        return results

    else:
        for (i,path) in enumerate(paths):
            trail = path.get("Last")
            match_kg = match_the_entity([trail])
            if len(match_kg) > 0:
                path["match_kg"] = match_kg

        return paths

def paths_package(cluster_chain_of_entities,chain_of_entities,depth):
    cluster_chain_of_entities_pack = []
    chain_of_entities_pack = []
    if len(cluster_chain_of_entities) > 0:
        nodeTemp = cluster_chain_of_entities
        for index in range(2, depth):
            chainsNode = []
            for node in nodeTemp:
                chainsNode.extend(node.children)
            nodeTemp = chainsNode

        for node in nodeTemp:
            for entity in chain_of_entities:
                header = entity["First"]
                if header in node.match_kg:
                    childNode = Node(firstMatch=entity.get("First"),
                                     lastMatch=entity.get("Last"),
                                     trail=entity.get("trail"),
                                     header=entity.get("header"),
                                     relation=entity.get("relation"),
                                     match_kg=entity.get("match_kg"),
                                     headRelation=entity.get("head"))
                    node.add_child(childNode)
        cluster_chain_of_entities_pack = cluster_chain_of_entities
        for node in cluster_chain_of_entities_pack:
            mpl = MultiPathLinkedList(node)
            paths = mpl.print_all_local_subpaths()
            if paths != None:
                chain_of_entities_pack.extend([path for path in paths if path not in chain_of_entities_pack])

    else:
        for chain in chain_of_entities:
            node = Node(firstMatch=chain.get("First"),
                        lastMatch=chain.get("Last"),
                        trail=chain.get("trail"),
                        header=chain.get("header"),
                        relation=chain.get("relation"),
                        match_kg=chain.get("match_kg"),
                        headRelation=chain.get("head"))

            chain_of_entities_pack.append(node.paths())
            cluster_chain_of_entities_pack.append(node)

    return cluster_chain_of_entities_pack, chain_of_entities_pack


def reasoning(question, cluster_chain_of_entities):
    response = reasoning_answer(question, cluster_chain_of_entities)
    res = changeStrToJson(response)
    if if_true(res):
        return True, response
    else:
        return False, response


def reasoning_answer(question, cluster_chain_of_entities):
    query = ANSWER_MEDICAL_EXAM_QUESTION_PROMPT.format(question=question, cluster_chain_of_entities = cluster_chain_of_entities)
    response = run_llm(query)
    return response


def if_true(prompt):
    if isinstance(prompt, dict):
        lowercase_data = {k.lower() for k, v in prompt.items()}
        if "yes" in lowercase_data:
            return True
        else:
            return False
    elif isinstance(prompt, str):
        if "yes" in prompt.lower():
            return True
        return False
    else:
        return False