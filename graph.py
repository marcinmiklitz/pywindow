#!/usr/bin/env python3
""" Module under construction. Creating molecular models and povray inputs. """

def molecule_to_povray():
    """
    print('\nNitrogen\n')
    for i in atom_list_translated2:
        if i[0] == 'N':
            print("// id {0}".format(i[1]))
            print("sphere{{<{2:.4f},{3:.4f},{4:.4f}>, {5}}}".format(*i, atomic_vdw_radius[i[0]]))
    print('\nCarbon\n')
    for i in atom_list_translated2:
        if i[0] == 'C':
            print("// id {0}".format(i[1]))
            print("sphere{{<{2:.4f},{3:.4f},{4:.4f}>, {5}}}".format(*i, atomic_vdw_radius[i[0]]))
    print('\nHydrogen\n')
    for i in atom_list_translated2:
        if i[0] == 'H':
            print("// id {0}".format(i[1]))
            print("sphere{{<{2:.4f},{3:.4f},{4:.4f}>, {5}}}".format(*i, atomic_vdw_radius[i[0]]))"""
    return

#This is the new part for the neighbours search and to create nice
#wireframe simplified CC3 figures
"""
def find_neighbours(elements, coordinates):
    output_dict = {}
    for i in range(len(elements)):
        dist_mat = euclidean_distances(coordinates[i].reshape(1, -1), coordinates)
        shell_2 = [x for x in np.where(dist_mat < 2)[1] if x != i]
        shell_4 = [x for x in np.where(dist_mat < 3)[1] if x != i]
        shell_6 = [x for x in np.where(dist_mat < 6)[1] if x != i]
        output = [elements[i],
                  [shell_2, [elements[x] for x in shell_2]],
                  [shell_4, [elements[x] for x in shell_4]],
                  [shell_6, [elements[x] for x in shell_6]],
                 ]
        if output[0] != 'H':
            output_dict[i] = output
    return(output_dict)

neighberoods = find_neighbours(mol.molecular_data['elements'],
                               mol.molecular_data['coordinates'])

def define_neighberhood(system):
    shell_1 = [system[1][1].count('C'), system[1][1].count('N'), system[1][1].count('H')]
    shell_2 = [system[2][1].count('C'), system[2][1].count('N'), system[2][1].count('H')]
    shell_3 = [system[3][1].count('C'), system[3][1].count('N'), system[3][1].count('H')]
    fingerprint = [system[0], shell_1, shell_2, shell_3]
    return(fingerprint)

fingerprints = {}
for i in neighberoods.keys():
    fingerprint = define_neighberhood(neighberoods[i])
    fingerprints[i] = fingerprint

def define_chemical_type(fingerprints):
    output_dict = {}
    for i in fingerprints.keys():
        if fingerprints[i][0] == 'C':
            if fingerprints[i][1][0] == 1 and fingerprints[i][1][1] == 1:
                #'imine carbon' 'CN'
                output_dict[i] = 'CN'
            if fingerprints[i][1][0] == 3 and fingerprints[i][1][1] == 0:
                #'aromatic carbon 1' 'CA1'
                output_dict[i] = 'CA1'
            if fingerprints[i][1][0] == 2 and fingerprints[i][1][1] == 0:
                if fingerprints[i][1][2] == 1:
                    #'aromatic carbon 2' 'CA2'
                    output_dict[i] = 'CA2'
                if fingerprints[i][1][2] == 2:
                    if fingerprints[i][2][1] == 0:
                        #'cyclohexane carbon 2 'CC2'
                        output_dict[i] = 'CC2'
                    if fingerprints[i][2][1] > 0:
                        #'cyclohexane carbon 3' 'CC3'
                        output_dict[i] = 'CC3'
            if fingerprints[i][1][0] == 2 and fingerprints[i][1][1] == 1:
                #'cyclohexane carbon 1' 'CC1'
                output_dict[i] = 'CC1'
    return(output_dict)

type_dict = define_chemical_type(fingerprints)

def find_benzene(type_dict, neighberoods):
    benzenes_dict = {}
    no_of_rings = 0
    alread_used = []
    for i in type_dict.keys():
        if i not in alread_used:
            if type_dict[i] in ['CA1', 'CA2']:
                benzenes_dict[no_of_rings] = [i]
                for step in range(2):
                    for j in benzenes_dict[no_of_rings]:
                        for k in neighberoods[j][1][0]:
                            if k in type_dict.keys():
                                if type_dict[k] in ['CA1', 'CA2']:
                                    if k not in benzenes_dict[no_of_rings]:
                                        benzenes_dict[no_of_rings].append(k)
                for used in benzenes_dict[no_of_rings]:
                    alread_used.append(used)
                no_of_rings += 1
    return(benzenes_dict)

benzenes = find_benzene(type_dict, neighberoods)
benzenes

def CC1_pairs(type_dict, neighberoods):
    CC1_dict = {}
    no_of_pairs = 0
    alread_used = []
    for i in type_dict.keys():
        if i not in alread_used:
            if type_dict[i] in ['CC1']:
                CC1_dict[no_of_pairs] = [i]
                for k in neighberoods[i][1][0]:
                    if k in type_dict.keys():
                        if type_dict[k] in ['CC1']:
                            CC1_dict[no_of_pairs].append(k)
                for used in CC1_dict[no_of_pairs]:
                    alread_used.append(used)
                no_of_pairs += 1
    return(CC1_dict)

CC1_paired = CC1_pairs(type_dict, neighberoods)
CC1_paired

def CC2_pairs(type_dict, neighberoods):
    CC2_dict = {}
    no_of_pairs = 0
    alread_used = []
    for i in type_dict.keys():
        if i not in alread_used:
            if type_dict[i] in ['CC2']:
                CC2_dict[no_of_pairs] = [i]
                for k in neighberoods[i][1][0]:
                    if k in type_dict.keys():
                        if type_dict[k] in ['CC2']:
                            CC2_dict[no_of_pairs].append(k)
                for used in CC2_dict[no_of_pairs]:
                    alread_used.append(used)
                no_of_pairs += 1
    return(CC2_dict)

CC2_paired = CC2_pairs(type_dict, neighberoods)
CC2_paired

molecules_framework = {'atom_ids': [], 'elements': [], 'coordinates': [], 'connect': []}

for i in benzenes.keys():
    coor_array = np.array([mol.molecular_data['coordinates'][x] for x in benzenes[i]])
    molecules_framework['atom_ids'].append('CA{0}'.format(i))
    molecules_framework['elements'].append('C')
    molecules_framework['coordinates'].append(center_of_coor(coor_array))
    molecules_framework['connect'].append([i+1])


for i in CC1_paired.keys():
    coor_array = np.array([mol.molecular_data['coordinates'][x] for x in CC1_paired[i]])
    molecules_framework['atom_ids'].append('CC{0}'.format(i))
    molecules_framework['elements'].append('C')
    cc1_com = center_of_coor(coor_array)
    molecules_framework['coordinates'].append(cc1_com)
    connect_to = euclidean_distances(cc1_com.reshape(1, -1), molecules_framework['coordinates'][:len(benzenes)])[0]
    for j in range(len(connect_to)):
        if connect_to[j] < 5.6:
            molecules_framework['connect'][j].append(len(benzenes) + i + 1)
    molecules_framework['connect'].append([len(benzenes)+ i +1])
    #for j in range(len(connect_to)):
    #    if connect_to[j] < 6:
    #        molecules_framework['connect'][-1].append(j)


for i in CC2_paired.keys():
    coor_array = np.array([mol.molecular_data['coordinates'][x] for x in CC2_paired[i]])
    molecules_framework['atom_ids'].append('O{0}'.format(i))
    molecules_framework['elements'].append('O')
    cc2_com = center_of_coor(coor_array)
    molecules_framework['coordinates'].append(cc2_com)
    connect_to = euclidean_distances(cc2_com.reshape(1, -1),
                                     molecules_framework['coordinates'][len(benzenes):len(benzenes)+len(CC1_paired)])[0]
    for j in range(len(connect_to)):
        if connect_to[j] < 3:
            molecules_framework['connect'][j + len(benzenes)].append(len(benzenes) + len(CC1_paired) + i + 1)
            #molecules_framework['connect'].append([len(benzenes)+ i + len(CC1_paired) + 1, j + len(benzenes)])

for i in range(len(window_network)):
    molecules_framework["atom_ids"].append('Si{}'.format(i + len(main_network)))
    molecules_framework["coordinates"] = np.append(molecules_framework["coordinates"], [window_network[i]], axis=0)
    molecules_framework["elements"].append('Si')

    connect_to = euclidean_distances(window_network[i].reshape(1, -1), main_network)[0]
    for j in range(len(connect_to)):
        if connect_to[j] < 4:
            if start_num + len(main_network) + i + 1 not in molecules_framework['connect'][start_connect + j]:
                molecules_framework['connect'][start_connect + j].append(start_num + len(main_network) + i + 1)

    molecules_framework['connect'].append([start_num + len(main_network) + i + 1]);

main_network = []
window_network = []
with open("CC3b_bulk_2017322_19271.txt") as file_src:
    file = [i.split() for i in file_src.readlines()[1:]]
    for i in file:
        main_network.append([float(i[4]), float(i[5]), float(i[6])])
        window_network.append([float(i[16]), float(i[17]), float(i[18])])
        window_network.append([float(i[21]), float(i[22]), float(i[23])])
        window_network.append([float(i[26]), float(i[27]), float(i[28])])
        window_network.append([float(i[31]), float(i[32]), float(i[33])])

main_network = np.array(main_network)
window_network = np.array(window_network)

start_num = len(molecules_framework["atom_ids"])
start_connect = len(molecules_framework["connect"])

for i in range(len(main_network)):
    molecules_framework["atom_ids"].append('Si{}'.format(i))
    molecules_framework["coordinates"] = np.append(molecules_framework["coordinates"], [main_network[i]], axis=0)
    molecules_framework["elements"].append('Si')
    molecules_framework["connect"].append([i + 1 + start_num])

molecules_framework["connect"][start_connect-10:];

for i in range(len(window_network)):
    connect_to = euclidean_distances(window_network[i].reshape(1, -1), window_network)[0]
    for j in range(len(connect_to)):
        if 10 < connect_to[j] < 11:
            molecules_framework['connect'][start_connect + len(main_network) + j].append(start_num + len(main_network) + i + 1)
        if 11.06 < connect_to[j] < 11.08:
            molecules_framework['connect'][start_connect + len(main_network) + j].append(start_num + len(main_network) + i + 1)


save = Output().frame2pdb(filename="CC3b_bulk_framework", atom_ids=molecules_framework["atom_ids"],
                         coordinates=molecules_framework["coordinates"],
                         elements=molecules_framework["elements"],
                         connect=molecules_framework['connect'])"""
