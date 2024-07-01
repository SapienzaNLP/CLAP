import penman
import re
from penman import loads
import amr
import constants
import random as rd
pat = re.compile(r"([)])")
pat2 = re.compile(r"([(])")

from penman.models.amr import model

model.reifications["contrast-01"] = [":concession", ":ARG1", ":ARG2"]
model.dereifications["contrast-01"] = [(":concession", ":ARG1", ":ARG2")]
model.dereifications["have-rel-role-91"] = [(":relate", ":ARG0", ":ARG1"),(":role-of", ":ARG2", ":ARG0"),(":role-of", ":ARG3", ":ARG1"),(":mode", ":ARG2", ":ARG4")]
model.dereifications["include-91"] = [(":subset", ":ARG2", ":ARG1")]


def dereify_edges(graph):
    dereify_graph_triples = graph.triples
    dereifications = {}
    add_triples = {}
    add_triples_2 = {}
    remove_triples = []

    replace_map = {}
    for (head, rel, tail) in graph.triples:
        if rel == ":instance" and tail in model.dereifications:
            dereifications = model.dereifications[tail]
            
            aux_remove_triples = [(head, rel, tail)]
            dereify_triple_relations = list(set([v for (h,r,t) in dereifications for v in (h,r,t) if v != h]))

            for dereify_triple in dereifications:

                new_head = None
                new_tail = None 
                inv = None
                non_reify = False
                is_main = True
                
                for (other_head, other_rel, other_tail) in graph.triples:
                    
                    if head == other_head and other_rel == dereify_triple[1]:
                        aux_remove_triples.append((other_head, other_rel, other_tail))
                        new_head = other_tail 
                    elif head == other_head and other_rel == dereify_triple[2]:
                        aux_remove_triples.append((other_head, other_rel, other_tail))
                        new_tail = other_tail


                    elif head == other_tail and other_rel.endswith("-of") and other_rel.replace("-of", "") == dereify_triple[1]:
                        is_main = False
                        aux_remove_triples.append((other_head, other_rel, other_tail))
                        new_head = other_head
                    elif head == other_tail and other_rel.endswith("-of") and other_rel.replace("-of", "") == dereify_triple[2]:
                        is_main = False
                        aux_remove_triples.append((other_head, other_rel, other_tail))
                        new_tail = other_head

                    if not inv and (new_head or new_tail):
                        inv = new_head if new_head else new_tail

                    if (head == other_head and other_rel not in dereify_triple_relations and other_rel != ":instance") or \
                        (head == other_tail and other_rel.endswith("-of") and other_rel.replace("-of", "") not in dereify_triple_relations):
                        non_reify = True
                        break
                
                if new_head and new_tail and not non_reify:
                    remove_triples.extend(aux_remove_triples)
                    if inv == new_head:
                        
                        add_triples.setdefault(new_tail, []).append((new_head, dereify_triple[0], new_tail))
                        add_triples_2.setdefault(aux_remove_triples[-1], []).append((new_head, dereify_triple[0], new_tail))
                        replace_map[head] = new_head
                    else:
                        new_rel = dereify_triple[0]+"-of" if dereify_triple[0] != ":subset" else ":superset"
                        add_triples.setdefault(new_head, []).append((new_tail, dereify_triple[0]+"-of", new_head))
                        replace_map[head] = new_tail
                        add_triples_2.setdefault(aux_remove_triples[-1], []).append((new_tail, new_rel, new_head))


                    for k, ts in add_triples_2.items():
                        new_t = []
                        for (n_h, n_r, n_t) in ts:
                            if k == head:
                                new_t.append((n_h, n_r, replace_map[head]))
                            else:
                                new_t.append((n_h, n_r, n_t))

                        add_triples_2[k] = new_t

    new_triples = []
    remove_triples = set(remove_triples)

    for k,v in replace_map.items():
        new_v = v
        while new_v in replace_map:
            new_v = replace_map[new_v]

        replace_map[k] = new_v

    # for (head, rel, tail) in graph.triples:
    #     if (head, rel, tail) in remove_triples:
    #         if tail in add_triples:
    #             triple_list = add_triples[tail]
    #             for h, r, t in triple_list:
    #                 new_triples.append((replace_map[h] if h in replace_map else h, r, replace_map[t] if t in replace_map else t))
    #         
    #         remove_triples.remove((head, rel, tail))
    #     else:
    #         new_triples.append((replace_map[head] if head in replace_map else head, rel, replace_map[tail] if tail in replace_map else tail))

    for (head, rel, tail) in graph.triples:
        if (head, rel, tail) in remove_triples:
            if (head, rel, tail) in add_triples_2:
                triple_list = add_triples_2[(head, rel, tail)]
                for h, r, t in triple_list:
                    new_triples.append((replace_map[h] if h in replace_map else h, r, replace_map[t] if t in replace_map else t))
            
            remove_triples.remove((head, rel, tail))
        else:
            new_triples.append((replace_map[head] if head in replace_map else head, rel, replace_map[tail] if tail in replace_map else tail))

    dereify_graph = penman.loads(penman.encode(penman.graph.Graph(new_triples)))[0]

    return dereify_graph

class BaseLinearization():
    
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.new_variable = 0
        self.variables = {}
        self.new_tokens_init = constants.new_tokens_remove_next_init
        self.new_tokens_end = ["-of"]
        self.pointer_span = "<p>"


    def __call__(self, x):
        return self._linearize(x)

    def linearize_reduced(self, encoded_graph):
        encoded_graph = penman.encode(encoded_graph)

        self.new_variable = 0
        self.variables = {}
        self.variables_inv = {}

        lin_graph = ""
        starts_graph = False
        for line in encoded_graph.splitlines():
            if line.startswith('('):
                starts_graph = True
            
            if starts_graph:        
                line = line.strip()
                
                for span in line.split(" "):
                    span = span.strip().split("~e.")[0]
                    if not span.startswith('"'):
                        lin_graph += pat2.sub(r"\1 ", pat.sub(r" \1", span)).strip() + " "
                    elif span.endswith('"'):
                        lin_graph += span + " "
                    else:
                        lin_graph += ('"'.join(span.split('"')[:-1]) + '"' + pat.sub(r" \1", span.split('"')[-1])).strip() + " "   
        

        # replace variable with #1, #2, #3, ...
        entity = False
        splitted_lin_graph = lin_graph.split(" ")
        for idx, span in enumerate(splitted_lin_graph):
            if span.startswith('"'):
                entity = True
            
            if span.endswith('"'):
                entity = False

            if not entity and span == "/":
                self.variables[splitted_lin_graph[idx-1]] = self._new_variable()
                self.variables_inv[self.variables[splitted_lin_graph[idx-1]]] = splitted_lin_graph[idx-1]

        new_var_graph = ""
        for idx, span in enumerate(splitted_lin_graph):
            if span not in  ["/"]:
                new_span = self.variables[span] if idx \
                        and (splitted_lin_graph[idx-1].startswith(":")  or splitted_lin_graph[idx-1] == "(") \
                        and span in self.variables else span

                new_var_graph += new_span + " " 
        new_graph = new_var_graph.strip()

        return new_graph




    def linearize_extreme(self, graph, wiki=False):

        if not wiki:
            graph.triples = [triple for triple in graph.triples if triple[1] != ":wiki"]
        
        # extract concepts from graph and add name entities with their name
        # input: [('n2', ':instance', 'need-01'), ('n2', ':ARG0', 's'), ('s', ':instance', 'state'), ('s', ':wiki', '"Texas"'), ('s', ':name', 'n'), ('n', ':instance', 'name'),  ('n', ':op1', '"Texas"'), ('s', ':part', 'c'), ('c', ':instance', 'city'), ('c', ':wiki', "Austin,_Texas"'), ('c', ':name', 'n3'), ('n3', ':instance', 'name'), ('n3', ':op1', '"Austin"'),('c', ':mod', 'e'),  ('e', ':instance', 'especially'), ('n2', ':ARG1', 'h'), ('h', ':instance', 'help-01'), ('h', ':ARG2', 's')]
        # output: ['need-01', 'state_Texas', 'city_Austin', 'especially', 'help-01']
        concepts = []
        concept_map = {}
        relations = []
        delete = set()
        
        url_set = set()
        url_entity_set = set()
        ord_entities = {}

        for triple in graph.triples:
            if triple[2] == "url-entity":
                    url_entity_set.add(triple[0])

            elif triple[1] == ":instance" and triple[2] != "name":
                concepts.append(triple[0])
                concept_map[triple[0]] = triple[2]

                if triple[2] == "hyperlink-91":
                    url_set.add(triple[0])
            # elif triple[1] == ":ord":
            #     ord_entities[triple[2]] = ""
            # elif triple[0] in ord_entities and triple[1] == ":value":
            #     ord_entities[triple[0]] = triple[2]

                

            # inverse url-entity triples
        url_switch = {}
        new_triple = []
        for triple in graph.triples:
            if triple[0] in url_entity_set and triple[1] == ":ARG3-of" and triple[2] in url_set:
                url_switch[triple[0]] = triple[2]
                url_switch[triple[2]] = triple[0]

        for triple in graph.triples:
            if triple[0] in url_switch and triple[1] == ":ARG3-of" and triple[2] in url_switch:
                new_triple.append((triple[2], ":ARG3", triple[0]))
                if graph.top == triple[0]:
                    graph.top = triple[2]
            elif triple[2] in url_switch:
                new_triple.append((triple[0], triple[1], url_switch[triple[2]]))

            elif triple[1] == ":ord" and triple[2] in ord_entities:
                new_triple.append((triple[0], triple[1], ord_entities[triple[2]]))
            elif triple[0] in ord_entities:
                continue
            else:
                new_triple.append(triple)
        
        graph.triples = new_triple

        #for triple in graph.triples:
        #    if triple[1] == ":instance" and triple[2] != "name":
        #        continue
        #    if triple[1] in [":quant", ":value"] or (triple[2].startswith('"') and triple[2].endswith('"') and not triple[1].startswith(":op")):
        #        value = triple[2].replace('"', '')
        #        concept_map[triple[0]] = concept_map[triple[0]] + f"#{triple[1][1:]}#" + value
        #        delete.add((triple[0], triple[1], triple[2]))
        
        url_entity_name = set()
        url_switch = {}
        names = {}
        delete_names = set()

        for triple in graph.triples:
            if triple[1] == ":name":
                names[triple[2]] = triple[0]
                delete.add((triple[0], triple[1], triple[2]))
                delete_names.add(triple[2])
            elif triple[0] in url_entity_set and triple[1] == ":value":
                delete.add((triple[0], triple[1], triple[2]))
                url_entity_name.add(triple[2])

            elif  triple[0] in url_entity_set and url_set:
                delete.add((triple[0], triple[1], triple[2]))

        for triple in graph.triples:
            if triple[0] in url_entity_name:
                delete.add((triple[0], triple[1], triple[2]))


        for triple in graph.triples:
            if triple[1] == ":op1" and triple[0] in names:
                concept_map[names[triple[0]]] = concept_map[names[triple[0]]] + " :name " + triple[2].replace('"', '')
                delete.add((triple[0], triple[1], triple[2]))                
            elif triple[1].startswith(":op") and triple[0] in names:
                concept_map[names[triple[0]]] = concept_map[names[triple[0]]] + "_" + triple[2].replace('"', '')
                delete.add((triple[0], triple[1], triple[2]))
            elif triple[0] in url_set and triple[1] in [":ARG3"] and triple[2] in url_entity_set:
                delete.add((triple[0], triple[1], triple[2]))
        
        new_triple = []
        for triple in graph.triples:
            if triple[0] in concept_map and triple[1] == ":instance":
                new_triple.append((triple[0], triple[1], concept_map[triple[0]]))
            elif (triple[0], triple[1], triple[2]) in delete or (triple[0] in delete_names and triple[1] == ":instance" and triple[2] == "name"):
                continue
            elif triple[1] not in [":name", ":wiki"] and (triple[2].startswith('"') and triple[2].endswith('"')):
                new_triple.append((triple[0], triple[1], triple[2] + "#v#"))
            else: 
                new_triple.append(triple)
        
        graph.triples = new_triple
        
        encoded_graph = penman.encode(graph)

        self.new_variable = 0
        self.variables = {}
        self.variables_inv = {}
        lin_graph = ""
        starts_graph = False
        for line in encoded_graph.splitlines():
            if line.startswith('('):
                starts_graph = True
            
            if starts_graph:        
                line = line.strip()
                
                for span in line.split(" "):
                    span = span.strip().split("~e.")[0]
                    if not span.startswith('"'):
                        span = span.replace("-of", "of") if span.startswith(":") else span
                        lin_graph += pat2.sub(r"\1 ", pat.sub(r" \1", span)).strip() + " "

                    elif span.endswith('"'):
                        lin_graph += span + " "
                    else:
                        lin_graph += ('"'.join(span.split('"')[:-1]) + '"' + pat.sub(r" \1", span.split('"')[-1])).strip() + " "   

        # replace variable with #1, #2, #3, ...
        entity = False
        splitted_lin_graph = lin_graph.split(" ")
        
        for idx, span in enumerate(splitted_lin_graph):
            if span.startswith('"'):
                entity = True
            
            if span.endswith('"'):
                entity = False

            if not entity and span == "/":
                self.variables[splitted_lin_graph[idx-1]] = self._new_variable()
                self.variables_inv[self.variables[splitted_lin_graph[idx-1]]] = splitted_lin_graph[idx-1]
        
        spans_repeated = set()
        # create a map from variable to concept
        variable_concept = {}
        for idx, span in enumerate(splitted_lin_graph):
            if idx and splitted_lin_graph[idx-1] == "/":
                if not span in spans_repeated:
                    variable_concept[splitted_lin_graph[idx-2]] = span
                    spans_repeated.add(span)
                else:
                    i = 2
                    while span + "#" + str(i).zfill(2) in spans_repeated:
                        i += 1
                    
                    spans_repeated.add(span + "#" + str(i).zfill(2))
                    variable_concept[splitted_lin_graph[idx-2]] = span + "#" + str(i).zfill(2)

        new_var_graph = ""

        for idx, span in enumerate(splitted_lin_graph):
            if idx and splitted_lin_graph[idx-1] in [":name", ":wiki"] and "_" in span:
                new_span = span.replace("_", " ").replace('"', "")
                new_var_graph += new_span + " " 
            
            elif idx and splitted_lin_graph[idx-1] == "/":
                new_span = variable_concept[splitted_lin_graph[idx-2]] if splitted_lin_graph[idx-2] in variable_concept else span
                new_var_graph += new_span + " " 

            elif span not in  ["(", "/"] and not (idx+1 < len(splitted_lin_graph) and splitted_lin_graph[idx+1] == "/"):
                new_span = variable_concept[span] + " )" if span in variable_concept else span
                new_var_graph += new_span.replace('"', "") + " " 

        # reduce ")" using number, for example ") ) ) )" -> ")3"

        parenthesis_count = 0
        new_graph = ""
        for idx, span in enumerate(new_var_graph.strip().split(" ")):
            if parenthesis_count and span == ")":
                parenthesis_count += 1

            else:
                if parenthesis_count > 1:
                    new_graph = new_graph.strip() + str(parenthesis_count-1) + " "
                    parenthesis_count = 0
                elif parenthesis_count == 1:
                    parenthesis_count = 0
                elif span == ")":
                    parenthesis_count += 1
                
                new_graph += span + " "

        new_graph = " " + new_graph.strip()


        new_triples = []
        for (head, rel, tail) in graph.triples:
            if ":name" in tail:
                new_triples.append((head, rel, tail.split(":name")[0].strip() + ' :name "' + tail.split(":name")[1].strip() + '"'))
            elif tail.endswith("#v#"):
                new_triples.append((head, rel, tail[:-3]))
            else:
                new_triples.append((head, rel, tail))
        
        graph.triples = new_triples

        try:
            graph = penman.loads(penman.encode(graph))[0]
        except:
            print(graph.metadata)
            print(graph.triples)
        graph.metadata["triples"] = []

        graph.triples_reduced = []

        spans_repeated = set()
        variable_concept = {}

        for (head, rel, tail) in graph.triples:
            if rel == ":instance" and head not in spans_repeated:
                variable_concept[head] = tail
                spans_repeated.add(tail)
            elif  rel == ":instance" :
                i = 2
                while tail + "#" + str(i).zfill(2) in spans_repeated:
                    i += 1
                
                spans_repeated.add(tail + "#" + str(i).zfill(2))
                variable_concept[head] = tail + "#" + str(i).zfill(2)

            elif rel == ":name":
                variable_concept[head] = tail.replace('"', "")


        graph.triples_reduced.append(("TOP", ":top", variable_concept[graph.triples[0][0]]))
        for (head, rel, tail) in graph.triples:
            if rel in [":name", ":wiki", ":instance"]:
                continue
            
            new_head, _, new_tail = head, rel, tail
            if head in variable_concept:
                new_head = variable_concept[head]
            
            if tail in variable_concept:
                new_tail = variable_concept[tail]
                
            graph.triples_reduced.append((new_head, rel, new_tail))

        graph.metadata["triples"] = str(graph.triples_reduced)

        return graph, new_graph


    def _new_variable(self):
        self.new_variable += 1
        return self.pointer_span+ str(self.new_variable)


    def decode_graph_reduced(self, linearized_graph):
        self.new_variable = 1000

        correct_graph = True
        open_parentensis = 0
        open_comma = False
        filtered_graph = []
        check_variable = {}
        splitted_lin_graph = linearized_graph.replace(":quantity", ":quantity ").replace(":quantity  ", ":quantity ").replace(":quantity _of", ":quantity_of").split(" ")

        for idx, span in enumerate(splitted_lin_graph):

            if span.startswith('"'):
                open_comma = True

            if span.endswith('"'):
                open_comma = False

            # add splitted new token (e.g. 0, 1, 2, etc) to the previous token (e.g. :ARG, :op, :snt, <p>)
            if filtered_graph and filtered_graph[-1] in self.new_tokens_init:
                filtered_graph[-1] += span
                continue

            # elif span.isnumeric() and filtered_graph and filtered_graph[-1].isnumeric():
            #     filtered_graph[-1] += span
            #     continue
            
            # filter out wrong token
            elif filtered_graph and filtered_graph[-1].startswith(self.pointer_span) and span == self.pointer_span:
                continue
            
            # add -of token to the previous token
            elif span in self.new_tokens_end:
                filtered_graph[-1] += span
                continue
            
            # add open parenthesis of node
            # elif span.startswith(self.pointer_span):
            elif span == "(":
                # filtered_graph.append("(")
                open_parentensis += 1

            # del open parenthesis of a variable
            elif len(filtered_graph) > 1 and filtered_graph[-1].startswith(self.pointer_span) and (span.startswith(":") or span == ")") and filtered_graph[-2] == "(":
                del filtered_graph[-2]
                open_parentensis -= 1

            # add slash between variable and node
            elif filtered_graph and filtered_graph[-1].startswith(self.pointer_span) and not (span.startswith(":") or span == ")") :
                filtered_graph.append("/")

                # check repeated variables
                if filtered_graph[-2] not in check_variable:
                    check_variable[filtered_graph[-2]] = filtered_graph[-2]
                else:
                    new_variable = self._new_variable()
                    check_variable[new_variable] = filtered_graph[-2]
                    filtered_graph[-2] = new_variable
                    correct_graph = False

            # check node structure
            elif len(filtered_graph) > 1 and filtered_graph[-2] == "/" and not (span.startswith(":") or span in ["(", ")"]):
                filtered_graph[-1] += span
                continue


            # check quotes
            elif open_parentensis > 0 and (span in ["(", ")"] or span.startswith(":")) and open_comma:
                filtered_graph[-1] += '"'
                open_comma = False
            
            # check wrong relations
            elif open_parentensis > 0 and span == "(" and not filtered_graph[-1].startswith(":"):
                filtered_graph.append(":wrong")
                correct_graph = False

            # check variable nodes
            elif  len(filtered_graph) > 1 and span == ")" and filtered_graph[-1].startswith(self.pointer_span) and filtered_graph[-2].startswith("("):
                filtered_graph.append("/")
                filtered_graph.append("wrong-node")
                correct_graph = False

            if span == ")":
                open_parentensis -= 1

                if open_parentensis < 1:
                    open_parentensis = 0
                    filtered_graph.append(span)
                    if idx + 1 < len(splitted_lin_graph):
                        correct_graph = False

                    break

            filtered_graph.append(span)

        # check number of parenthesis
        while open_parentensis > 0:
            filtered_graph.append(")")
            open_parentensis -= 1
            correct_graph = False

        lin_graph = " ".join(filtered_graph).replace(self.pointer_span, "z").strip()

        try: 
            if not amr.AMR.parse_AMR_line(lin_graph):
                print(linearized_graph)
                print(lin_graph)
                return ("( z0 / bark-01 :ARG0 ( z1 / dog ) :ARG1 ( z2 / tree ) )",  False)

            graph = loads(lin_graph)
        except:
            print(linearized_graph)
            print(lin_graph)
            return ("( z0 / bark-01 :ARG0 ( z1 / dog ) :ARG1 ( z2 / tree ) )",  False)
            
        return (lin_graph, correct_graph)

        

    def decode_graph_extreme_triples(self, linearized_graph, wiki=False):
        splitted_lin_graph = linearized_graph.strip().replace(":quant", ":quant ").replace(":quant  ", ":quant ").replace(":quant of", ":quantof").split(" ")


        try:
            splitted_graph = []
            # connect name entities: (e.g. ":name John Smith" -> ":name John_Smith")
            open_parenthesis = False
            for idx, span in enumerate(splitted_lin_graph):
                if len(splitted_graph) > 1 and splitted_graph[-2] in [":name", ":wiki", ":value"] and not (span.startswith(":") or (span == ")" and not open_parenthesis) or (span[0] == ")" and span[1:].isnumeric())):
                    if span == "(":
                        open_parenthesis = True
                    elif span == ")":
                        open_parenthesis = False

                    splitted_graph[-1] += "_" + span
                
                elif  splitted_graph and ((splitted_graph[-1] in [":snt", ":op", ")"] and span.isnumeric()) or (splitted_graph[-1].startswith(":") and span == "of")):
                    splitted_graph[-1] += span

                else:
                    splitted_graph.append(span)

            # print(splitted_lin_graph)
            # print(splitted_graph)
            splitted_lin_graph = splitted_graph
            new_variable = 0
            concept_ids_map = {}
            triples_list = []
            previous_id = []
            id_map = {}

            for idx, span in enumerate(splitted_lin_graph):
                
                # if span.startswith(':snt'):
                #     while len(previous_id) > 1 and id_map[previous_id[-1]] != "multi-sentence":
                #         previous_id.pop()
                
                if span.startswith(")") and  not (span[1:] and not span[1:].isnumeric()):
                    parenthesis_counter = int(span[1:]) if span[1:] else 0
                    while parenthesis_counter:
                        if not previous_id:
                            break
                        previous_id.pop()
                        parenthesis_counter -= 1

                    if not previous_id:
                        break

                    previous_id.pop()
                    if not previous_id:
                        break
                
                elif span == "12:" or span == "5:":
                    rel = splitted_lin_graph[idx-1]
                    triples_list.append((previous_id[-1], rel, span))

                elif triples_list and triples_list[-1][1] in [":quant", ":time"] and triples_list[-1][2].endswith(":"):
                    new_triplet = (triples_list[-1][0], triples_list[-1][1], triples_list[-1][2][:-1] + span + "'")
                    triples_list[-1].pop()
                    triples_list.append(new_triplet)

                elif idx and (span in ["+", "-"] or (splitted_lin_graph[idx-1] in [":polarity", ":mode", ":polite"] and span in ["expressive", "imperative"])):
                    triples_list.append((previous_id[-1], splitted_lin_graph[idx-1], span))

                elif not span.startswith(':') and span.endswith('#v#'):
                    new_span = '"' + span[:-3] + '"'
                    rel = splitted_lin_graph[idx-1]
                    triples_list.append((previous_id[-1], rel, new_span))

                elif not span.startswith(':') and span.endswith('v>'):
                    new_span = '"' + span[:-2] + '"'
                    rel = splitted_lin_graph[idx-1]
                    triples_list.append((previous_id[-1], rel, new_span))


                elif splitted_lin_graph and splitted_lin_graph[idx-1] in [":quant", ":time", ":li", ":value"] and not span.startswith(':') and ":" in span:
                    new_span = '"' + span + '"'
                    rel = splitted_lin_graph[idx-1]
                    triples_list.append((previous_id[-1], rel, new_span))


                elif not span.startswith(':') and span.replace(".", "").isnumeric():
                    new_span = span 
                    rel = splitted_lin_graph[idx-1]
                    triples_list.append((previous_id[-1], rel, new_span))




                elif idx and splitted_lin_graph[idx-1] == ":wiki":
                    triples_list.append((previous_id[-1], ":wiki", '"' + span + '"'))
                
                elif not span.startswith(':') and "/" in span:
                    triples_list.append((previous_id[-1], splitted_lin_graph[idx-1], '"' + span + '"'))

                    
                elif not span.startswith(':') and not (idx and splitted_lin_graph[idx-1] == ':name'):
                    new_id = "z" + str(new_variable) if span not in concept_ids_map else concept_ids_map[span]
                    
                    if idx and previous_id:
                        rel = splitted_lin_graph[idx-1][:-2] + "-of" if splitted_lin_graph[idx-1].endswith("of") else splitted_lin_graph[idx-1]
                        # rel = splitted_lin_graph[idx-1]
                        triples_list.append((previous_id[-1], rel, new_id))
                    # elif not previous_id:
                    #     break
                    previous_id.append(new_id)
                    id_map[new_id] = span.split("#")[0]
                    if span not in concept_ids_map:
                        concept_ids_map[span] = new_id
                        concept_ids_map[new_id] = span
                        triples_list.append((new_id, ":instance", span.split("#")[0]))
                        new_variable += 1
                
                elif idx and splitted_lin_graph[idx-1] == ":name":
                    new_id = "z" + str(new_variable) 
                    new_variable += 1
                    if not wiki and False:
                        triples_list.append((previous_id[-1], ":wiki", "-"))
                    triples_list.append((previous_id[-1], ":name", new_id))
                    triples_list.append((new_id, ":instance", "name"))
                    id_map[new_id] = span.split("#")[0]

                    names = span.split("_")
                    op = 1

                    for name in names:
                        triples_list.append((new_id, f":op{op}", f'"{name}"'))
                        op += 1

            lin_graph = penman.encode(penman.graph.Graph(triples_list))

            g = loads(lin_graph)

            try:
                if not amr.AMR.parse_AMR_line(lin_graph):
                    return ("(z0 / bark-01 :ARG0 (z1 / dog))",  False)
            except:
                return ("(z0 / bark-01 :ARG0 (z1 / dog))",  False)
            
            return (lin_graph, True)
        except:
            return ("(z0 / bark-01 :ARG0 (z1 / dog))",  False)

        
        
    def __repr__(self):
        return self.__class__.__name__ + str(self._args) + str(self._kwargs)

    def __str__(self):
        return self.__class__.__name__ + str(self._args) + str(self._kwargs)

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self._args == other._args and
                self._kwargs == other._kwargs)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.__class__, self._args, self._kwargs))

