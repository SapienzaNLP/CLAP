import torch
import numpy as np
import constants
import math
import ast

def shift_tokens_left(input_ids, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.copy()

    for i in range(len(input_ids)):
        shifted_input_ids[i] = np.roll(input_ids[i], -1)
        shifted_input_ids[i][-1] = -100

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
    

    return shifted_input_ids

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
    
def warmup_embeddings(model, tokenizer):
    # get input embeddings
    embeddings = model.get_input_embeddings()

    with torch.no_grad():

        for tok, idx in tokenizer.get_added_vocab().items():
            vecs = []

            tok = tok.lstrip(tokenizer.init_token).lstrip(" ")
            
            tok_split = [tok]
                
            for tokk in tok_split:
                for tok_id in tokenizer.encode(tokk, add_special_tokens=False):
                    vecs.append(embeddings.weight[tok_id].clone())


            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, +0.1)
                vec += noise
                embeddings.weight[idx] = vec

    model.set_input_embeddings(embeddings)
    model.tie_weights()



def get_alignment_matrix(snt_ids, graph_ids, alignments, tokenizer):
    # create batch alignment tensor, size = len(enumerate(sentences)) x len(input_ids) x len(decoder_input_ids)
    batch_alignment = torch.zeros(len(snt_ids), len(graph_ids))

    # delete padding graph ids
    graph_ids = [g if g != -100 else 1 for g in graph_ids]
    graph_tokens = [tok.replace(" ", "Ġ") for tok in tokenizer.convert_ids_to_tokens(graph_ids)]

    alignment_map = {}
    for alignment in alignments.split():
            alignment_map.setdefault(int(alignment.split("-")[0]), []).append(alignment.split("-")[1])

    node_pos = ""
    current_node_pos = 1
    node_pos_stack = []
    target_node_map = {}
    node_pos = str(current_node_pos)

    node_pos_stack.append(current_node_pos)
    current_node_pos = 1

    for token_idx, token in enumerate(graph_tokens):
        if token.startswith(f"{tokenizer.init_token}<p>"):
            next_token_idx = token_idx + 2

            if not graph_tokens[next_token_idx].startswith(f"{tokenizer.init_token}:") and not graph_tokens[next_token_idx].startswith(f"{tokenizer.init_token})"):
                target_node_map.setdefault(node_pos, []).append(next_token_idx)

            elif not graph_tokens[next_token_idx].startswith(f"{tokenizer.init_token})"):
                node_pos += "." + str(current_node_pos)
                target_node_map.setdefault(node_pos, []).append(token_idx)
                current_node_pos += 1
                node_pos = ".".join(node_pos.split(".")[:-1])
                                
            else:
                node_pos += "." + str(current_node_pos)
                target_node_map.setdefault(node_pos, []).append(token_idx)
                node_pos = ".".join(node_pos.split(".")[:-1])
                current_node_pos = node_pos_stack.pop()
                current_node_pos += 1

        elif  graph_tokens[next_token_idx].startswith(f"{tokenizer.init_token})"):
            current_node_pos = node_pos_stack.pop()
            current_node_pos += 1  

        elif token.startswith(f"{tokenizer.init_token}:"):
            next_token_idx = token_idx + 1

            while not graph_tokens[next_token_idx].startswith(f"{tokenizer.init_token}"):
                next_token_idx += 1

            if graph_tokens[next_token_idx].startswith(f"{tokenizer.init_token}<p>"):
                node_pos += "." + str(current_node_pos)
                target_node_map.setdefault(node_pos + ".r", []).append(token_idx)


    input_word_pos_map = {}
    pos = 0
    sentence_tokenized = tokenizer.convert_ids_to_tokens(snt_ids)

    for word_idx, word in enumerate(sentence_tokenized):
        if word.startswith(f"{tokenizer.init_token}") and word != f"{tokenizer.init_token}<s>" and word != f"{tokenizer.init_token}</s>" and not (word == f"{tokenizer.init_token}<" and (word_idx + 1) < len (sentence_tokenized) and sentence_tokenized[word_idx + 1] == "a"):
            if pos in alignment_map:
                for node_position in alignment_map[pos]:
                    if node_position in target_node_map:
                        batch_alignment[word_idx][target_node_map[node_position]] = 1

            pos += 1

        elif not word.startswith(f"{tokenizer.init_token}"):
            input_word_pos_map[word_idx] = pos

    return batch_alignment

def sequence_infilling(input_ids, pad_token_id, mask_token_id, mlm_prob=0.35, device="cuda"):
    res_ids = []

    for ids in input_ids:
        # inp_ids = torch.LongTensor([iid for iid in ids if iid != pad_token_id and iid != -100]).to(device)
        token_length = len([int(itm != pad_token_id) for itm in ids])
        masking_length = math.floor(token_length * mlm_prob)
        masked_length = 1
        masked_inputs = ids.copy()
        while masked_length < masking_length:
            span_length = min(math.floor(np.random.poisson(3, 1)), token_length - 1)
            start_index = math.floor(np.random.uniform(1, token_length - span_length, 1))
            # masked_inputs = torch.cat((masked_inputs[:start_index], torch.tensor([mask_token_id]).to(device), masked_inputs[start_index + span_length:]), dim=0)
            masked_inputs = masked_inputs[:start_index] + [mask_token_id] + masked_inputs[start_index + span_length:]
            token_length -= span_length - 1
            masked_length += span_length
        res_ids.append(masked_inputs)

    return res_ids

def jointly_infilling(snts_ids, graphs_ids, alignments, mask_token_id, tokenizer, mlm_prob=0.35, device="cuda"):
    res_ids = []

    for idx, snt_ids in enumerate(snts_ids):
        token_length = len(snt_ids)
        graph_ids = graphs_ids[idx]
        alignment_matrix = get_alignment_matrix(snt_ids, graph_ids, alignments[idx], tokenizer)

        masking_length = math.floor(token_length * mlm_prob)
        masked_length = 1
        masked_inputs = input_id.copy()
        while masked_length < masking_length:
            span_length = min(math.floor(np.random.poisson(3, 1)), token_length - 1)
            start_index = math.floor(np.random.uniform(1, token_length - span_length, 1))
            masked_inputs = masked_inputs[:start_index] + [mask_token_id] + masked_inputs[start_index + span_length:]
            token_length -= span_length - 1
            masked_length += span_length
        res_ids.append(masked_inputs)

    return res_ids

def sequence_infilling_batch(input_ids, pad_token_id, mask_token_id, mlm_prob=0.35, device="cuda"):
    res_ids = []

    for ids in input_ids:
        inp_ids = torch.LongTensor([iid for iid in ids if iid != pad_token_id and iid != -100]).to(device)
        token_length = len([int(itm != pad_token_id) for itm in ids])
        masking_length = math.floor(token_length * mlm_prob)
        masked_length = 1
        masked_inputs = inp_ids.clone()
        while masked_length < masking_length:
            span_length = min(math.floor(np.random.poisson(3, 1)), token_length - 1)
            start_index = math.floor(np.random.uniform(1, token_length - span_length, 1))
            masked_inputs = torch.cat((masked_inputs[:start_index], torch.tensor([mask_token_id]).to(device), masked_inputs[start_index + span_length:]), dim=0)
            # masked_inputs = masked_inputs[:start_index] + [mask_token_id] + masked_inputs[start_index + span_length:]
            token_length -= span_length - 1
            masked_length += span_length
        res_ids.append(masked_inputs)

    return res_ids

def permute_cross_attn(cross_attn):

    cross_tensor = [torch.stack(decoder_tok, dim=0) for decoder_tok in cross_attn]
 
    return torch.stack(cross_tensor, dim=0).permute(4, 2, 1, 3, 0, 5).squeeze(dim=0)


def permute_cross_attn_forward(cross_attn, model_name):
    # transform list of tensors (cross_attn) to tensor
    cross_tensor = torch.stack(cross_attn, dim=0)
    # if "t5" in model_name.lower():
    #     # T5 models: cross_tensor shape is (num_layers, batch_size, seq_length, seq_length)
    #     # Permute to (batch_size, num_layers, seq_length, seq_length)
    #     # print shape
# 
    #     print(cross_tensor.shape)
    #     exit()
    #     return cross_tensor.permute(0, 1, 2, 3, 4)
    # 
    # elif "bart" in model_name:
    #     return cross_tensor.permute(1, 0, 2, 3, 4)
    # else:
        
    return cross_tensor.permute(1, 0, 2, 3, 4)



# build graph maps from tokenized graph
def build_graph_maps(graph_tokens, init_token, init_pos = 1):
    new_tokens = constants.new_tokens_remove_next_init
    # create a map that aligned tokenized graph to the original graphs
    current_node_pos = 1
    node_pos_stack = []

    target_node_map = {}
    node_pos = str(current_node_pos)
    target_node_map[init_pos] = node_pos
    node_pos_stack.append(current_node_pos)


    reentrancy_map = {}
    non_reentrancy_map = {'<p>1':"1"}
    named_entities_map = {}

    is_lit = False
    is_named_entity = False
    name_entity_root = None
    prev_wiki = False
    
    
    for token_idx, token in enumerate(graph_tokens):
        if token == f'{init_token}"':
            is_lit = True
        elif token == f'"':
            is_lit = False

        if not is_lit and token == f"{init_token})":
            is_named_entity = False
            name_entity_root = None

        if  token == f"{init_token}:wiki":
            current_node_pos += 1
            prev_wiki = True
        elif not is_lit:
            if token.startswith(f"{init_token}:"):
                next_token_idx = token_idx + 1
                try:
                    while graph_tokens[next_token_idx-1].lstrip(init_token) in new_tokens or not graph_tokens[next_token_idx].startswith(init_token) or graph_tokens[next_token_idx] == f'{init_token}"' or graph_tokens[next_token_idx] == f'{init_token}(':
                        next_token_idx += 1
                
                except:
                    print(graph_tokens)
                    exit()


            if token.startswith(f"{init_token}:") and graph_tokens[next_token_idx].startswith(f"{init_token}<p>"):
                next_next_token_idx = next_token_idx + 2

                while not graph_tokens[next_next_token_idx].startswith(init_token) or graph_tokens[next_next_token_idx] == f'{init_token}"':
                    next_next_token_idx += 1

                if not (graph_tokens[next_next_token_idx].startswith(f"{init_token}:") or graph_tokens[next_next_token_idx].startswith(f"{init_token})")):

                    if token.startswith(f"{init_token}:name"):
                        is_named_entity = True
                        name_entity_root = node_pos + "." + str(current_node_pos)
                        if not (current_node_pos == 2 and not prev_wiki):
                            named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                        else: 
                            named_entities_map.setdefault(name_entity_root, []).append(node_pos)

                        
                        prev_wiki = False

                    node_pos += "." + str(current_node_pos)
                    current_node_pos += 1
                    node_pos_stack.append(current_node_pos)
                    current_node_pos = 1
                    token = graph_tokens[next_token_idx]
                    target_node_map[next_token_idx] = node_pos
                    target_node_map[token_idx] = node_pos + ".r"

                    non_reentrancy_map[graph_tokens[next_token_idx].replace(f"{init_token}", "")] = node_pos

                    if is_named_entity:                    
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos + ".r")

                else:
                    node_pos += "." + str(current_node_pos)

                    target_node_map[next_token_idx] = node_pos
                    target_node_map[token_idx] = node_pos + ".r"
                    
                    if is_named_entity:
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos + ".r")

                    if graph_tokens[next_token_idx].startswith(f"{init_token}<p>"):
                        reentrancy_map[node_pos] = graph_tokens[next_token_idx].replace(f"{init_token}", "")

                    current_node_pos += 1
                    node_pos = ".".join(node_pos.split(".")[:-1])

        

            elif  (token.startswith(f"{init_token})") or token.startswith(")")) and node_pos_stack:
                node_pos = ".".join(node_pos.split(".")[:-1])
                current_node_pos = node_pos_stack.pop()

    return target_node_map, reentrancy_map, non_reentrancy_map, named_entities_map


# build graph maps from tokenized graph
def build_graph_extreme_maps(graph_tokens, init_token, init_pos = 2):
    # create a map that aligned tokenized graph to the original graphs
    current_node_pos = 1
    node_pos_stack = [1]

    target_node_map = {}
    node_pos = str(current_node_pos)
    prev_string = False


    for token_idx, token in enumerate(graph_tokens):

        if token_idx -1 > 0 and (token.replace(f"{init_token}", "").replace("_", "").isnumeric() and (graph_tokens[token_idx-1].startswith(f"{init_token}:op") or graph_tokens[token_idx-1].startswith(f"{init_token}:snt") or graph_tokens[token_idx-1].startswith(f"{init_token})"))):
            continue
        if prev_string and (token.startswith(f"{init_token}:") or token.startswith(f"{init_token})")):
            prev_string = False
        elif prev_string:
            continue
        
        i = 1
        while token_idx+i < len(graph_tokens) and not graph_tokens[token_idx+i].startswith(f"{init_token}"):
            i += 1

        is_string = False
        j = i
        while token_idx+j < len(graph_tokens) and not (graph_tokens[token_idx+j].startswith(f"{init_token}:") or graph_tokens[token_idx+j].startswith(f"{init_token})")):
            if graph_tokens[token_idx+j] == "#" and graph_tokens[token_idx+j+1] == "v" and graph_tokens[token_idx+j+2] == "#":
                is_string = True
                break
            j += 1
        
        if  token.startswith(f"{init_token}:") and token not in [f"{init_token}:op",f"{init_token}:snt", f"{init_token})"] and (token in [f"{init_token}:wiki", f"{init_token}:name", f"{init_token}:value", f"{init_token}:mode", f"{init_token}:polite"] \
                                                    or is_string \
                                                    or (token_idx+1 < len(graph_tokens) and graph_tokens[token_idx+1] in ["-", "+"]) \
                                                    or (token_idx+2 < len(graph_tokens) and graph_tokens[token_idx+1] == init_token and graph_tokens[token_idx+2] in ["-", "+"]) \
                                                    or graph_tokens[token_idx+i].startswith(f"{init_token}") and (graph_tokens[token_idx+i].replace(f"{init_token}", "").replace("_", "").isnumeric() or graph_tokens[token_idx+i].replace(f"{init_token}", "").replace("_", "") in ["-", "+"])):
            prev_string = True
            node_pos += "." + str(current_node_pos)

            target_node_map[token_idx + 1] = node_pos
            target_node_map[token_idx] = node_pos + ".r"
        
            current_node_pos += 1
            node_pos = ".".join(node_pos.split(".")[:-1])


        elif token.startswith(f"{init_token})"):

            node_pos = ".".join(node_pos.split(".")[:-1])
            current_node_pos = node_pos_stack.pop() 
            if token_idx + 1 < len(graph_tokens):
                next_token = graph_tokens[token_idx + 1].replace(f"{init_token}", "").replace("_", "")
                if next_token.isnumeric():
                    next_token = int(next_token)
                    while next_token:
                        node_pos = ".".join(node_pos.split(".")[:-1])
                        current_node_pos = node_pos_stack.pop() 
                        next_token -= 1

        elif token.startswith(f"{init_token}:") or (token.startswith(":") and graph_tokens[token_idx-1] == f"{init_token}"):
            node_pos += "." + str(current_node_pos)
            target_node_map[token_idx] = node_pos + ".r"
            current_node_pos += 1
            node_pos_stack.append(current_node_pos)
            current_node_pos = 1

        elif (token.startswith(f"{init_token}") and not token == init_token) or (graph_tokens[token_idx-1] == f"{init_token}" and token not in [":", ")"]):
            target_node_map[token_idx] = node_pos


    return target_node_map

     

# build graph maps from tokenized graph
def build_graph_bmr_maps(graph_tokens, init_token, init_pos = 2):
    new_tokens = constants.new_tokens_remove_next_init
    # create a map that aligned tokenized graph to the original graphs
    current_node_pos = 1
    node_pos_stack = []

    target_node_map = {}
    node_pos = str(current_node_pos)
    target_node_map[init_pos] = node_pos
    node_pos_stack.append(current_node_pos)
    current_node_pos = 1

    reentrancy_map = {}
    non_reentrancy_map = {'<p>1':"1"}
    named_entities_map = {}

    is_lit = False
    is_named_entity = False
    name_entity_root = None
    prev_wiki = False
    
    
    for token_idx, token in enumerate(graph_tokens):
        if token == f'{init_token}"':
            is_lit = True
        elif token == f'"':
            is_lit = False

        if not is_lit and token == f"{init_token})":
            is_named_entity = False
            name_entity_root = None

        elif not is_lit:
            if token.startswith(f"{init_token}:"):
                next_token_idx = token_idx + 1
                try:
                    while graph_tokens[next_token_idx-1].lstrip(init_token) in new_tokens or not graph_tokens[next_token_idx].startswith(init_token) or graph_tokens[next_token_idx] == f'{init_token}"':
                        next_token_idx += 1
                
                except:
                    print(graph_tokens)
                    exit()

            if token.startswith(f"{init_token}:") and graph_tokens[next_token_idx].startswith(f"{init_token}<p>"):
                next_next_token_idx = next_token_idx + 2
                while not graph_tokens[next_next_token_idx].startswith(init_token) or graph_tokens[next_next_token_idx] == f'{init_token}"':
                    next_next_token_idx += 1

                if not (graph_tokens[next_next_token_idx].startswith(f"{init_token}:") or graph_tokens[next_next_token_idx].startswith(f"{init_token})")):

                    if token.startswith(f"{init_token}:name"):
                        is_named_entity = True
                        name_entity_root = node_pos + "." + str(current_node_pos)

                        named_entities_map.setdefault(name_entity_root, []).append(node_pos)


                    node_pos += "." + str(current_node_pos)
                    current_node_pos += 1
                    node_pos_stack.append(current_node_pos)
                    current_node_pos = 1
                    token = graph_tokens[next_token_idx]
                    target_node_map[next_token_idx] = node_pos
                    target_node_map[token_idx] = node_pos + ".r"

                    non_reentrancy_map[graph_tokens[next_token_idx].replace(f"{init_token}", "")] = node_pos

                    if is_named_entity:                    
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos + ".r")

                else:
                    node_pos += "." + str(current_node_pos)

                    target_node_map[next_token_idx] = node_pos
                    target_node_map[token_idx] = node_pos + ".r"
                    
                    if is_named_entity:
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos)
                        named_entities_map.setdefault(name_entity_root, []).append(node_pos + ".r")

                    if graph_tokens[next_token_idx].startswith(f"{init_token}<p>"):
                        reentrancy_map[node_pos] = graph_tokens[next_token_idx].replace(f"{init_token}", "")

                    current_node_pos += 1
                    node_pos = ".".join(node_pos.split(".")[:-1])

        

            elif  token.startswith(f"{init_token})") and node_pos_stack:
                node_pos = ".".join(node_pos.split(".")[:-1])
                current_node_pos = node_pos_stack.pop()
    
    return target_node_map, reentrancy_map, non_reentrancy_map, named_entities_map

    
def extract_alignment_bart(cross_attn, snt_ids, amr_ids, predictions_status, tokenizer):
    alignment_map = []
    init_token = tokenizer.init_token

    for sentence_idx in range(len(cross_attn)):
        if not predictions_status[sentence_idx]:
            alignment_map.append("")
            continue

        shift = 0 if snt_ids[sentence_idx][0] else 1

        alignment_score = np.squeeze(cross_attn[sentence_idx].detach().cpu().to(torch.float16).numpy())
        # alignment_score = np.squeeze(cross_attn[sentence_idx].detach().cpu().numpy())

        graph_tokens = [token if not token.startswith(" ") else token.replace(" ", init_token) for token in tokenizer.convert_ids_to_tokens(amr_ids[sentence_idx])]
        graph_tokens = graph_tokens[:-2]
        sentence_tokenized = tokenizer.convert_ids_to_tokens(snt_ids[sentence_idx])

        input_word_pos_map = {}
        pos = 0
        for word_idx, word in enumerate(sentence_tokenized):
            if word.startswith(init_token) and not (word == f"{init_token}<" and (word_idx + 1) < len(sentence_tokenized) and sentence_tokenized[word_idx + 1] == "a"):
                pos += 1

            input_word_pos_map[word_idx] = pos

            if word == "<s>":
                pos += 1
                    
        target_node_map, _, _, _ = build_graph_maps(graph_tokens, init_token)

        # remove score from stop words from graph and wikinodes
        stop_words_graph =  [f'{init_token})', '<pad>', '<s>', '</s>', f' :wiki', f'{init_token}"', "en_XX", "es_XX", "fr_FR", "it_IT", "de_DE"]   
        is_lit = False
        is_wiki = False

        for graph_token_idx, graph_token in enumerate(graph_tokens):
            if graph_token == f"{init_token}:wiki":
                is_wiki = True
            elif graph_token == f'{init_token}"':
                is_lit = True
            elif graph_token == '"':
                is_lit = False
                is_wiki = False
            elif graph_token == f'{init_token}-' and graph_tokens[graph_token_idx - 1] == f"{init_token}:wiki":
                is_wiki = False
                alignment_score[:, :, graph_token_idx, :] = 0 


            if graph_token in stop_words_graph or (is_wiki and is_lit):
                alignment_score[:, :, graph_token_idx, :] = 0 


        stop_words_input = ["en_XX", "es_XX", "fr_FR", "it_IT", "de_DE", '<s>', '</s>', '<pad>', '<pad>', f'{init_token}-', f'{init_token},', f'{init_token}@', f'{init_token}.', ".", f'{init_token}:']
        for snt_token_idx, snt_token in enumerate(sentence_tokenized):
            if snt_token in stop_words_input:
                alignment_score[:, :, :, snt_token_idx] = 0

        # identify compound tokens in the sentence and sum the values
        sentence_tokens_filter = [(token_idx, 1) if token_idx > 1 else (token_idx, 0) for token_idx, token in enumerate(sentence_tokenized) if not token.startswith(init_token) and token_idx > 1 and token not in ['<s>', '</s>']]
        sentence_tokens_map = {}
        for token_idx, repeated in sentence_tokens_filter:
            sentence_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in sentence_tokens_map \
                                                            else sentence_tokens_map[token_idx - 1]
        # ccreate 1 array of lenght of sentence with 1s
        length_compound_tokens = np.ones(len(alignment_score[0, 0, 0,:]))

        for split_token_idx, repeated in sentence_tokens_filter:
            alignment_score[:, :, :, sentence_tokens_map[split_token_idx]] += alignment_score[:, :, :, split_token_idx]
            alignment_score[:, :, :, split_token_idx] = 0
            length_compound_tokens[sentence_tokens_map[split_token_idx]] += 1
            length_compound_tokens[split_token_idx] = 1

        # extract sentence word related to word position to token in sentence
        # sentence_words_map = {}
        # for encoder_pos, sentence_token_pos in input_word_pos_map.items():
        #     sentence_word = sentence_tokenized[encoder_pos].replace(f"{init_token}", "")
        #     next_token = 1
        #     while (encoder_pos + next_token) < len(sentence_tokenized) and not sentence_tokenized[encoder_pos + next_token].startswith(init_token):
        #         sentence_word += sentence_tokenized[encoder_pos + next_token]
        #         next_token += 1
# 
        #     sentence_words_map[sentence_token_pos] = sentence_word


        alignment_score = alignment_score[8:].sum(axis=0).sum(axis=0)
        # alignment_score = alignment_score[0:4].sum(axis=0).sum(axis=0)


        # create map relate node position to graph token
        graph_id_map = {}
        graph_nodes_map = {}
        pos2alignment_map = {}
        for graph_idx, graph_token in target_node_map.items():
            next_token = 2 if graph_idx + 2 < len(graph_tokens) and graph_tokens[graph_idx].startswith(f"{init_token}<p>") and not (graph_tokens[graph_idx + 1].startswith(f"{init_token}:") or graph_tokens[graph_idx + 1] == f"{init_token})") else 0
            if graph_idx + next_token < len(graph_tokens):
                graph_id = graph_tokens[graph_idx].replace(f"{init_token}", "")
                graph_node = graph_tokens[graph_idx + next_token].replace(f"{init_token}", "")
                
                # copy tensor
                sum_alignments = alignment_score[graph_idx + next_token, :].copy()
                
                next_token += 1
                is_prep_edge = (graph_idx + next_token) < len(graph_tokens) and graph_tokens[graph_idx + next_token] == f"{init_token}prep"

                while (graph_idx + next_token) < len(graph_tokens) \
                        and (not graph_tokens[graph_idx + next_token].startswith(init_token) \
                            or (is_prep_edge and (not graph_tokens[graph_idx + next_token].startswith(init_token) \
                                or graph_tokens[graph_idx + next_token] == f"{init_token}prep")
                            or (graph_tokens[graph_idx].startswith(f"{init_token}<p>") and graph_id != graph_node and not (graph_tokens[graph_idx + next_token].startswith(f"{init_token}:") or graph_tokens[graph_idx + next_token] == f"{init_token})")))):

                    graph_node += graph_tokens[graph_idx + next_token].lstrip(f"{init_token}")
                    if graph_tokens[graph_idx + next_token] != f"{init_token}-":
                        sum_alignments += alignment_score[graph_idx + next_token, :].copy()
                    
                    next_token += 1


                
                graph_id_map[graph_token] = graph_id
                graph_nodes_map[graph_token] = graph_node

                # if all the element in tensor are 0
                if np.sum(sum_alignments) != 0:
                    pos2alignment_map[graph_token] = sum_alignments


        node_word_pos_map = {}
        for node, alignment in pos2alignment_map.items():
            node_word_pos_map[node] = input_word_pos_map[np.argmax(alignment)]

        isi_alignments = []
        for node_pos,span  in node_word_pos_map.items():
            isi_alignments.append((int(span), node_pos))
        
        # sort alignments by second element
        isi_alignments = sorted(isi_alignments, key=lambda x: x[0])

        alignment_map.append(" ".join(f"{str(pos-shift)}-{node_pos}" for pos, node_pos in isi_alignments))

    
    return alignment_map


def extract_alignment_mbart_text(cross_attn, snt_ids, amr_ids, predictions_status, tokenizer):
    alignment_map = []
    init_token = tokenizer.init_token

    for sentence_idx in range(len(cross_attn)):
        if not predictions_status[sentence_idx]:
            alignment_map.append("")
            continue

        shift = 0 if snt_ids[sentence_idx][0] else 1

        alignment_score = np.squeeze(cross_attn[sentence_idx].detach().cpu().numpy())

        graph_tokens = [token if not token.startswith(" ") else token.replace(" ", init_token) for token in tokenizer.convert_ids_to_tokens(amr_ids[sentence_idx])]
        graph_tokens = graph_tokens[:-2]
        sentence_tokenized = tokenizer.convert_ids_to_tokens(snt_ids[sentence_idx])

        input_word_pos_map = {}
        pos = -1
        for word_idx, word in enumerate(sentence_tokenized):
            if  word == "<s>" or word.startswith(init_token) and not (word == f"{init_token}<" and (word_idx + 1) < len(sentence_tokenized) and sentence_tokenized[word_idx + 1] == "a"):
                pos += 1
            
            input_word_pos_map[word_idx] = pos
    
        target_node_map, _, _, _ = build_graph_maps(graph_tokens, init_token, init_pos=0)


        # remove score from stop words from graph and wikinodes
        stop_words_graph =  [f'{init_token})', '<pad>', '<s>', '</s>', f' :wiki', f'{init_token}"', "en_XX", "es_XX", "fr_FR", "it_IT", "de_DE"]   
        is_lit = False
        is_wiki = False

        for graph_token_idx, graph_token in enumerate(graph_tokens):
            if graph_token == f"{init_token}:wiki":
                is_wiki = True
            elif graph_token == f'{init_token}"':
                if is_lit:
                    is_lit = False
                    is_wiki = False
                else:
                    is_lit = True
            elif graph_token == f'{init_token}-' and graph_tokens[graph_token_idx - 1] == f"{init_token}:wiki":
                is_wiki = False
                alignment_score[:, :, graph_token_idx, :] = 0 


            if graph_token in stop_words_graph or (is_wiki and is_lit):
                alignment_score[:, :, graph_token_idx, :] = 0 


        stop_words_input = ["en_XX", "es_XX", "fr_FR", "it_IT", "de_DE", '<s>', '</s>', '<pad>', '<pad>', f'{init_token}-', f'{init_token},', f'{init_token}@', f'{init_token}.', ".", f'{init_token}:']
        for snt_token_idx, snt_token in enumerate(sentence_tokenized):
            if snt_token in stop_words_input:
                alignment_score[:, :, :, snt_token_idx] = 0
                

        # identify compound tokens in the sentence and sum the values
        sentence_tokens_filter = [(token_idx, 1) for token_idx, token in enumerate(sentence_tokenized) if not token.startswith(init_token) and token not in ['<s>', '</s>']]
        sentence_tokens_map = {}
        for token_idx, repeated in sentence_tokens_filter:
            sentence_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in sentence_tokens_map \
                                                            else sentence_tokens_map[token_idx - 1]
        # ccreate 1 array of lenght of sentence with 1s
        length_compound_tokens = np.ones(len(alignment_score[0, 0, 0,:]))


        for split_token_idx, repeated in sentence_tokens_filter:
            alignment_score[:, :, :, sentence_tokens_map[split_token_idx]] += alignment_score[:, :, :, split_token_idx]
            alignment_score[:, :, :, split_token_idx] = 0
            length_compound_tokens[sentence_tokens_map[split_token_idx]] += 1
            length_compound_tokens[split_token_idx] = 1

        # extract sentence word related to word position to token in sentence
        sentence_words_map = {}
        for encoder_pos, sentence_token_pos in input_word_pos_map.items():
            sentence_word = sentence_tokenized[encoder_pos].replace(f"{init_token}", "")
            next_token = 1
            while (encoder_pos + next_token) < len(sentence_tokenized) and not sentence_tokenized[encoder_pos + next_token].startswith(init_token):
                sentence_word += sentence_tokenized[encoder_pos + next_token]
                next_token += 1

            sentence_words_map[sentence_token_pos] = sentence_word


        # alignment_score = alignment_score[8:].sum(axis=0).sum(axis=0)
        alignment_score = alignment_score[0:4].sum(axis=0).sum(axis=0)

        # copy analignment score to zeros
        alignment_score_zeros = np.zeros((len(alignment_score), len(alignment_score[0])))
        graph_unit_node_map = {}
        new_pos_idx = 0
        for node_idx, _ in enumerate(graph_tokens):
            if node_idx in target_node_map:
                new_pos_idx = node_idx
                alignment_score_zeros[new_pos_idx] = alignment_score[node_idx]
            
            else:
                alignment_score_zeros[new_pos_idx] += alignment_score[node_idx]
            
            graph_unit_node_map[node_idx] = new_pos_idx


        node_word_pos_map = {}
        for token_pos, word in input_word_pos_map.items():
            higher_alignment = alignment_score_zeros[:, token_pos].copy()
            if np.sum(higher_alignment) != 0:
                node_word_pos_map[word] = target_node_map[graph_unit_node_map[np.argmax(higher_alignment)]]
    

        isi_alignments = []
        for span, node_pos  in node_word_pos_map.items():
            isi_alignments.append((int(span), node_pos))
        
        # sort alignments by second element
        isi_alignments = sorted(isi_alignments, key=lambda x: x[0])

        alignment_map.append(" ".join(f"{str(pos-shift)}-{node_pos}" for pos, node_pos in isi_alignments))

    
    return alignment_map



def extract_alignment_extrem_text(cross_attn, snt_ids, amr_ids, predictions_status, ids, tokenizer):
    alignment_map = []
    init_token = tokenizer.init_token
    content_words = ["JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS", "RB" , "RBR", "RBS", "NNP", "NNPS"]
    for sentence_idx in range(len(cross_attn)):
        if not predictions_status[sentence_idx]:
            alignment_map.append("")
            continue
        
        metadata = tokenizer.graphs_meta[ids[sentence_idx].item()]
        meta = {}
        for line in metadata.split("\n"):
            if line:
                key = line.strip().split()[1][2:]
                value = " ".join(line.strip().split()[2:])
                meta[key] = value
        
        metadata = meta
        
        shift = 0 if snt_ids[sentence_idx][0] else 1

        alignment_score = np.squeeze(cross_attn[sentence_idx].detach().cpu().float().numpy())

        graph_tokens = [token if not token.startswith(" ") else token.replace(" ", init_token) for token in tokenizer.convert_ids_to_tokens(amr_ids[sentence_idx])]
        graph_tokens = ["<s>"] + graph_tokens[:-2]
        sentence_tokenized = tokenizer.convert_ids_to_tokens(snt_ids[sentence_idx])

        input_word_pos_map = {}
        pos = -1
        for word_idx, word in enumerate(sentence_tokenized):
            if  word == "<s>" or word == " )" or word.startswith(init_token) and not (word == f"{init_token}<" and (word_idx + 1) < len(sentence_tokenized) and sentence_tokenized[word_idx + 1] == "a"):
                pos += 1
            
            input_word_pos_map[word_idx] = pos

        input_pos_word_map = {}

        for key, value in input_word_pos_map.items():
            input_pos_word_map.setdefault(value, []).append(key)


        target_node_map = build_graph_extreme_maps(graph_tokens, init_token, init_pos=0)


        alignment_score[:, :, :, 0] = 0


        # remove score from stop words from graph and wikinodes
        stop_words_graph =  [f'{init_token})', '<pad>', '<s>', '</s>', f' :wiki', f'{init_token}"', "en_XX", "es_XX", "fr_FR", "it_IT", "de_DE"]   
        is_lit = False
        is_wiki = False
    
    
        for graph_token_idx, graph_token in enumerate(graph_tokens):
            if graph_token == f"{init_token})":
                    alignment_score[:, :, graph_token_idx, :] = 0 
                    if graph_token_idx + 1< len(graph_tokens) and graph_tokens[graph_token_idx+1].isnumeric():
                        alignment_score[:, :, graph_token_idx + 1, :] = 0 
            elif graph_token in stop_words_graph:
                alignment_score[:, :, graph_token_idx, :] = 0 
            
            
            elif graph_token_idx in target_node_map and target_node_map[graph_token_idx][-1] == "r":
                alignment_score[:, :, graph_token_idx, :] = 0 


        stop_words_input = ["en_XX", "es_XX", "fr_FR", "it_IT", "de_DE", '<s>', '</s>', '<pad>', '<pad>', f'{init_token}-', f'{init_token},', f'{init_token}@', f'{init_token}.', ".", f'{init_token}:']
        for snt_token_idx, snt_token in enumerate(sentence_tokenized):
            if snt_token in stop_words_input:
                alignment_score[:, :, :, snt_token_idx] = 0
                

        # identify compound tokens in the sentence and sum the values
        sentence_tokens_filter = [(token_idx, 1) for token_idx, token in enumerate(sentence_tokenized) if not token.startswith(init_token) and token not in ['<s>', '</s>']]
        sentence_tokens_map = {}
        for token_idx, repeated in sentence_tokens_filter:
            sentence_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in sentence_tokens_map \
                                                            else sentence_tokens_map[token_idx - 1]
        # ccreate 1 array of lenght of sentence with 1s
        length_compound_tokens = np.ones(len(alignment_score[0, 0, 0,:]))


        # extract sentence word related to word position to token in sentence
        sentence_words_map = {}
        for encoder_pos, sentence_token_pos in input_word_pos_map.items():
            sentence_word = sentence_tokenized[encoder_pos].replace(f"{init_token}", "")
            next_token = 1
            while (encoder_pos + next_token) < len(sentence_tokenized) and not sentence_tokenized[encoder_pos + next_token].startswith(init_token):
                sentence_word += sentence_tokenized[encoder_pos + next_token]
                next_token += 1

            sentence_words_map[sentence_token_pos] = sentence_word


        alignment_score = alignment_score[:4].sum(axis=0).sum(axis=0)


        # copy analignment score to zeros
        alignment_score_zeros = np.zeros((len(alignment_score), len(alignment_score[0])))
        graph_unit_node_map = {}
        new_pos_idx = 0
        for node_idx, _ in enumerate(graph_tokens):
            if node_idx in target_node_map:
                new_pos_idx = node_idx
                alignment_score_zeros[new_pos_idx] = alignment_score[node_idx]
            
            else:
                alignment_score_zeros[new_pos_idx] += alignment_score[node_idx]
            
            graph_unit_node_map[node_idx] = new_pos_idx


        node_word_pos_map = {}


        for graph_idx, pos in target_node_map.items():
            higher_alignment = alignment_score_zeros[graph_idx, :].copy()
            if np.sum(higher_alignment) != 0:
                node_word_pos_map[pos] = input_word_pos_map[np.argmax(higher_alignment)]
        
        isi_alignments = []
        for node_pos, span  in node_word_pos_map.items():
            isi_alignments.append((int(span), node_pos))

        # pos_list = [1 if  pos_tag in content_words and not (lemma in ["be", "have"]) else 0 for lemma, pos_tag in zip(ast.literal_eval(metadata["lemmas"]), ast.literal_eval(metadata["pos"]))]


        # sort alignments by second element
        isi_alignments = sorted(isi_alignments, key=lambda x: x[0])


        # metadata_spans = ast.literal_eval(metadata["spans"])
# 
        # node_map = {}
        # isi_map = {}
# 
        # for pos, node_pos in isi_alignments:
        #     if node_pos not in node_map:
        #         node_map[node_pos] = pos
        #     else:
        #         isi_map[pos] = node_map[node_pos]
# 
        # merged_clusters = []
        # clusters_map = {}
        # for idx, spans in enumerate(metadata_spans):
        #     span_to_change = ""
        #     for span in spans:
        #         if span in isi_map:
        #             span_to_change = span
# 
        #     if span_to_change and span_to_change in isi_map and isi_map[span_to_change] in clusters_map:
        #         pos_cluster = clusters_map[isi_map[span_to_change]]
        #         merged_clusters[pos_cluster].extend(spans)
# 
        #     else: 
        #         pos_cluster = len(merged_clusters)
        #         merged_clusters.append(spans)
        #     
        #     for span in spans:
        #         clusters_map[span] = pos_cluster
# 
        #         
        # metadata["spans"] = str(merged_clusters)
    
        x_metadata = ""
        for key, value in metadata.items():
            x_metadata += "# ::" + key + " " + value + "\n"

        tokenizer.graphs_meta[ids[sentence_idx].item()] = x_metadata

        # alignment_map.append(" ".join(f"{str(pos-shift)}-{node_pos}" for pos, node_pos in isi_alignments))
        alignment_map.append(" ".join(f"{str(pos-0)}-{node_pos}" for pos, node_pos in isi_alignments))

    
    return alignment_map




def extract_alignment_mbart_text_graph(cross_attn, snt_ids, amr_ids, predictions_status, tokenizer):
    alignment_map = []
    init_token = tokenizer.init_token

    for sentence_idx in range(len(cross_attn)):
        if not predictions_status[sentence_idx]:
            alignment_map.append("")
            continue

        shift = 0 if snt_ids[sentence_idx][0] else 1

        alignment_score = np.squeeze(cross_attn[sentence_idx].detach().cpu().numpy())

        graph_tokens = [token if not token.startswith(" ") else token.replace(" ", init_token) for token in tokenizer.convert_ids_to_tokens(amr_ids[sentence_idx])]
        graph_tokens = graph_tokens[:-2]
        sentence_tokenized = tokenizer.convert_ids_to_tokens(snt_ids[sentence_idx])

        input_word_pos_map = {}
        pos = -1
        for word_idx, word in enumerate(sentence_tokenized):
            if  word == "<s>" or word.startswith(init_token) and not (word == f"{init_token}<" and (word_idx + 1) < len(sentence_tokenized) and sentence_tokenized[word_idx + 1] == "a"):
                pos += 1
            
            input_word_pos_map[word_idx] = pos
    
        target_node_map, _, _, _ = build_graph_maps(graph_tokens, init_token, init_pos=0)


        # remove score from stop words from graph and wikinodes
        stop_words_graph =  [f'{init_token})', '<pad>', '<s>', '</s>', f' :wiki', f'{init_token}"', "en_XX", "es_XX", "fr_FR", "it_IT", "de_DE"]   
        is_lit = False
        is_wiki = False

        for graph_token_idx, graph_token in enumerate(graph_tokens):
            if graph_token == f"{init_token}:wiki":
                is_wiki = True
            elif graph_token == f'{init_token}"':
                if is_lit:
                    is_lit = False
                    is_wiki = False
                else:
                    is_lit = True
            elif graph_token == f'{init_token}-' and graph_tokens[graph_token_idx - 1] == f"{init_token}:wiki":
                is_wiki = False
                alignment_score[:, :, graph_token_idx, :] = 0 


            if graph_token in stop_words_graph or (is_wiki and is_lit):
                alignment_score[:, :, graph_token_idx, :] = 0 


        stop_words_input = ["en_XX", "es_XX", "fr_FR", "it_IT", "de_DE", '<s>', '</s>', '<pad>', '<pad>', f'{init_token}-', f'{init_token},', f'{init_token}@', f'{init_token}.', ".", f'{init_token}:']
        for snt_token_idx, snt_token in enumerate(sentence_tokenized):
            if snt_token in stop_words_input:
                alignment_score[:, :, :, snt_token_idx] = 0
                

        # identify compound tokens in the sentence and sum the values
        sentence_tokens_filter = [(token_idx, 1) for token_idx, token in enumerate(sentence_tokenized) if not token.startswith(init_token) and token not in ['<s>', '</s>']]
        sentence_tokens_map = {}
        for token_idx, repeated in sentence_tokens_filter:
            sentence_tokens_map[token_idx] = token_idx - 1 if token_idx - 1 not in sentence_tokens_map \
                                                            else sentence_tokens_map[token_idx - 1]
        # ccreate 1 array of lenght of sentence with 1s
        length_compound_tokens = np.ones(len(alignment_score[0, 0, 0,:]))


        for split_token_idx, repeated in sentence_tokens_filter:
            alignment_score[:, :, :, sentence_tokens_map[split_token_idx]] += alignment_score[:, :, :, split_token_idx]
            alignment_score[:, :, :, split_token_idx] = 0
            length_compound_tokens[sentence_tokens_map[split_token_idx]] += 1
            length_compound_tokens[split_token_idx] = 1

        # extract sentence word related to word position to token in sentence
        sentence_words_map = {}
        for encoder_pos, sentence_token_pos in input_word_pos_map.items():
            sentence_word = sentence_tokenized[encoder_pos].replace(f"{init_token}", "")
            next_token = 1
            while (encoder_pos + next_token) < len(sentence_tokenized) and not sentence_tokenized[encoder_pos + next_token].startswith(init_token):
                sentence_word += sentence_tokenized[encoder_pos + next_token]
                next_token += 1

            sentence_words_map[sentence_token_pos] = sentence_word


        alignment_score = alignment_score[8:].sum(axis=0).sum(axis=0)
        # alignment_score = alignment_score[0:4].sum(axis=0).sum(axis=0)


        # create map relate node position to graph token
        graph_id_map = {}
        graph_nodes_map = {}
        pos2alignment_map = {}
        for graph_idx, graph_token in target_node_map.items():
            next_token = 2 if graph_idx + 2 < len(graph_tokens) and graph_tokens[graph_idx].startswith(f"{init_token}<p>") and not (graph_tokens[graph_idx + 1].startswith(f"{init_token}:") or graph_tokens[graph_idx + 1] == f"{init_token})") else 0
            if graph_idx + next_token < len(graph_tokens):
                graph_id = graph_tokens[graph_idx].replace(f"{init_token}", "")
                graph_node = graph_tokens[graph_idx + next_token].replace(f"{init_token}", "")
                
                # copy tensor
                sum_alignments = alignment_score[graph_idx + next_token, :].copy()


                next_token += 1
                is_prep_edge = (graph_idx + next_token) < len(graph_tokens) and graph_tokens[graph_idx + next_token] == f"{init_token}prep"

                while (graph_idx + next_token) < len(graph_tokens) \
                        and (not graph_tokens[graph_idx + next_token].startswith(init_token) \
                            or (is_prep_edge and (not graph_tokens[graph_idx + next_token].startswith(init_token) \
                                or graph_tokens[graph_idx + next_token] == f"{init_token}prep")
                            or (graph_tokens[graph_idx].startswith(f"{init_token}<p>") and graph_id != graph_node and not (graph_tokens[graph_idx + next_token].startswith(f"{init_token}:") or graph_tokens[graph_idx + next_token] == f"{init_token})")))):

                    graph_node += graph_tokens[graph_idx + next_token].lstrip(f"{init_token}")
                    if graph_tokens[graph_idx + next_token] != f"{init_token}-":
                        sum_alignments += alignment_score[graph_idx + next_token, :].copy()
                    
                    next_token += 1


                graph_id_map[graph_token] = graph_id
                graph_nodes_map[graph_token] = graph_node

                # if all the element in tensor are 0
                if np.sum(sum_alignments) != 0:
                    pos2alignment_map[graph_token] = sum_alignments



        node_word_pos_map = {}
        span_node_map = {}
        for node, alignment in pos2alignment_map.items():
            node_word_pos_map[node] = input_word_pos_map[np.argmax(alignment)]
            span_node_map.setdefault(np.argmax(alignment), []).append((np.max(alignment), node))

        for span_pos, nodes in span_node_map.items():
            if len(nodes) > 1:
                # sort by first element biggest to smallest
                nodes = sorted(nodes, key=lambda x: x[0], reverse=True)
                for node in nodes[1:]:
                    alignment_score_aux = pos2alignment_map[node[1]]
                    alignment_score_aux[span_pos] = 0
                    find = False
                    while not find and np.sum(alignment_score_aux) != 0:
                        pos_aux = np.argmax(alignment_score_aux)
                        if pos_aux not in span_node_map:
                            node_word_pos_map[node[1]] = input_word_pos_map[pos_aux]
                            find = True
                        else:
                            alignment_score_aux[pos_aux] = 0


        isi_alignments = []
        for node_pos,span  in node_word_pos_map.items():
            isi_alignments.append((int(span), node_pos))
        
        # sort alignments by second element
        isi_alignments = sorted(isi_alignments, key=lambda x: x[0])

        alignment_map.append(" ".join(f"{str(pos-shift)}-{node_pos}" for pos, node_pos in isi_alignments))

    return alignment_map

    