from pymysql import connect
from pandas import DataFrame
from numpy import zeros, int64, int32, float64, float32, multiply, dot, identity, sum
from itertools import permutations
from pickle import dump, load


db_name = 'binary_acm'


db = db_name
connection = connect(host="rcg-cs-ml-dev.dcr.sfu.ca", user="admin", password="joinbayes", db=db)
cursor = connection.cursor()
db_setup = db_name + "_setup"
connection_setup = connect(host="rcg-cs-ml-dev.dcr.sfu.ca", user="admin", password="joinbayes", db=db_setup)
cursor_setup = connection_setup.cursor()
db_bn = db_name + "_BN"
connection_bn = connect(host="rcg-cs-ml-dev.dcr.sfu.ca", user="admin", password="joinbayes", db=db_bn)
cursor_bn = connection_bn.cursor()


keys = {}


cursor_setup.execute("SELECT TABLE_NAME FROM EntityTables");
entity_tables = cursor_setup.fetchall()
entities = {}
for i in entity_tables:
    cursor.execute("SELECT * FROM " + i[0])
    rows = cursor.fetchall()
    cursor.execute("SHOW COLUMNS FROM " + db + "." + i[0])
    columns = cursor.fetchall()
    entities[i[0]] = DataFrame(rows, columns=[columns[j][0] for j in range(len(columns))])
    cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = " + "'" + i[0] + "'")
    key = cursor_setup.fetchall()
    keys[i[0]] = key[0][0]
    
    


cursor_setup.execute("SELECT TABLE_NAME FROM RelationTables  ")
relation_tables = cursor_setup.fetchall()
relations = {}
for i in relation_tables:
    cursor.execute("SELECT * FROM " + i[0])
    rows = cursor.fetchall()
    cursor.execute("SHOW COLUMNS FROM " + db + "." + i[0])
    columns = cursor.fetchall()
    relations[i[0]] = DataFrame(rows, columns=[columns[j][0] for j in range(len(columns))])
    cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = " + "'" + i[0] + "'")
    key = cursor_setup.fetchall()
    keys[i[0]] = key[0][0], key[1][0]
    
    


relation_names = tuple(i[0] for i in relation_tables)


indices = {}
for i in entity_tables:
    cursor_setup.execute("SELECT COLUMN_NAME FROM EntityTables WHERE TABLE_NAME = '" + i[0] + "'")
    key = cursor_setup.fetchall()[0][0]
    indices[key] = {}
    for index, row in entities[i[0]].iterrows():
        indices[key][row[key]] = index
        
        

matrices = {}
for i in relation_tables:
    cursor_setup.execute("SELECT REFERENCED_TABLE_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = " + "'" + i[0] + "'")
    reference = cursor_setup.fetchall()
    matrices[i[0]] = zeros((len(entities[reference[0][0]].index), len(entities[reference[1][0]].index)))
    
    
for i in relation_tables:
    cursor_setup.execute("SELECT COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = '" + i[0] + "'")
    key = cursor_setup.fetchall()
    cursor_setup.execute("SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM ForeignKeyColumns WHERE TABLE_NAME = '" + i[0] + "'")
    reference = cursor_setup.fetchall()
    for index, row in relations[i[0]].iterrows():
        matrices[i[0]][indices[reference[0][1]][row[key[0][0]]]][indices[reference[1][1]][row[key[1][0]]]] = 1
        
        
        
cursor_setup.execute("SELECT COLUMN_NAME, TABLE_NAME FROM AttributeColumns")
attribute_columns = cursor_setup.fetchall()
attributes = {}
for i in attribute_columns:
    attributes[i[0]] = i[1]
    
    
    
cursor_bn.execute("SELECT DISTINCT child FROM final_path_bayesnets_view")
childs = cursor_bn.fetchall()
rules = []
multiples = []
states = []
functors = {}
variables = {}
nodes = {}
masks = {}
base_indices = []
mask_indices = []
sort_indices = []
stack_indices = []
values = []
for i in range(len(childs)):
    rule = [childs[i][0]]
    cursor_bn.execute("SELECT parent FROM final_path_bayesnets_view WHERE child = " + "'" + childs[i][0] + "'")
    parents = cursor_bn.fetchall()
    for j in parents:
        if j[0] != '':
            rule += [j[0]]
    rules.append(rule)
    if len(rule) == 1:
        multiples.append(0)
    else:
        multiples.append(1)
    relation_check = 0
    for j in rule:
        if j.find(',') != -1:
            relation_check = 1
    functor = {}
    variable = {}
    node = {}
    state = []
    mask = {}
    unmasked_variables = []
    for j in range(len(rule)):
        fun = rule[j].split('(')[0]
        functor[j] = fun
        if rule[j].find(',') == -1:
            var = rule[j].split('(')[1][:-1]
            variable[j] = var
            node[j] = var[:-1]
            if relation_check == 0:
                unmasked_variables.append(var)
                state.append(0)
            else:
                mas = []
                for k in rule:
                    func = k.split('(')[0]
                    if func not in relation_names:
                            func = attributes[func]
                    if k.find(',') != -1 and k.find(var) != -1:
                        unmasked_variables.append(k.split('(')[1][:-1])
                        mas.append([func, k.split('(')[1].split(',')[0], k.split('(')[1].split(',')[1][:-1]]) 
                mask[j] = mas
                state.append(1)
        else:
            unmasked_variables.append(rule[j].split('(')[1][:-1])
            if fun in relation_names:
                state.append(2)
            else:
                state.append(3)     
    functors[i] = functor
    variables[i] = variable
    nodes[i] = node
    states.append(state)
    masks[i] = mask
    masked_variables = [unmasked_variables[0]]
    base_indice = [0]
    mask_indice = []
    for j in range(1, len(unmasked_variables)):
        mask_check = 0
        for k in range(len(masked_variables)):
            if unmasked_variables[j] == unmasked_variables[k]:
                mask_indice.append([k, j])
                mask_check = 1
        if mask_check == 0:
            base_indice.append(j)
            masked_variables.append(unmasked_variables[j])
    sort_indice = []
    sorted_variables = []
    if relation_check == 0:
        sort_indice.append([False, 0])
        sorted_variables.append(masked_variables[0])
    else:
        indices_permutations = list(permutations(range(len(masked_variables))))
        variables_permutations = list(permutations(masked_variables))
        for j in range(len(variables_permutations)):
            indices_chain = []
            variables_chain = []
            first = variables_permutations[j][0].split(',')[0]
            second = variables_permutations[j][0].split(',')[1]
            indices_chain.append([False, indices_permutations[j][0]])
            variables_chain.append(variables_permutations[j][0])
            untransposed_check = 1
            transposed_check = 1
            if len(variables_permutations[j]) > 1:
                for k in range(1, len(variables_permutations[j])):
                    next_first = variables_permutations[j][k].split(',')[0]
                    next_second = variables_permutations[j][k].split(',')[1]
                    if second == next_first:
                        second = next_second
                        indices_chain.append([False, indices_permutations[j][k]])
                        variables_chain.append(next_first + ',' + next_second)
                    elif second == next_second:
                        second = next_first
                        indices_chain.append([True, indices_permutations[j][k]])
                        variables_chain.append(next_second + ',' + next_first)    
                    else:
                        untransposed_check = 0
                        break
                if untransposed_check != 1:
                    indices_chain[0] = [True, indices_permutations[j][0]]
                    variables_chain[0] = second + ',' + first
                    temp = first
                    first = second
                    second = temp
                    for k in range(1, len(variables_permutations[j])):
                        next_first = variables_permutations[j][k].split(',')[0]
                        next_second = variables_permutations[j][k].split(',')[1]
                        if second == next_first:
                            second = next_second
                            indices_chain.append([False, indices_permutations[j][k]])
                            variables_chain.append(next_first + ',' + next_second)
                        elif second == next_second:
                            second = next_first
                            indices_chain.append([True, indices_permutations[j][k]])
                            variables_chain.append(next_second + ',' + next_first)    
                        else:
                            transposed_check = 0
                            break
            if untransposed_check == 1 or transposed_check == 1 or len(variables_permutations[j]) == 1:
                sort_indice = indices_chain
                sorted_variables = variables_chain
                break    
    stack_indice = []
    for j in range(1, len(sorted_variables)):
        second = sorted_variables[j].split(',')[1]
        for k in range(j - 1, -1, -1):
            previous_first = sorted_variables[k].split(',')[0]
            if previous_first == second:
                stack_indice.append([k, j])   
    base_indices.append(base_indice)
    mask_indices.append(mask_indice)
    sort_indices.append(sort_indice)
    stack_indices.append(stack_indice)
    cursor_bn.execute("SELECT * FROM `" + childs[i][0] + "_cp`")
    value = cursor_bn.fetchall()
    values.append(value)
    
    
dump([rules, values, states, entities, nodes, functors, multiples, indices, keys, masks, variables, matrices, attributes, relations, base_indices, mask_indices, sort_indices, stack_indices], open(db_name + "_data.pkl", "wb"))
