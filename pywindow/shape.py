#!/usr/bin/env python3
""" Module under construction """

"""
def window_analysis_with_shape(window, atom_list, atom_coor, atom_vdw, window_no, eps_sqrt):
    #print('\n    Length of the vextors for the window {0}'.format(len(window)))
    #print(window[:2])
    max_value = max(np.array(window)[:,1])
    number = [a for a, j in enumerate(window) if j[1] == max_value]
    vector2analyse = window[number[0]][5:8]
    vector_analysed = vector_analysis(vector2analyse, atom_list, atom_coor, atom_vdw, 0.1)
    #print('\n    The analysed most centered vetor')
    #print(vector_analysed)

    #UPDATE: Try rotation first, than translation
    vector_main = np.array([vector_analysed[5],vector_analysed[6],vector_analysed[7]])
    vec_a = [1, 0, 0]
    #vec_b = [0, 1, 0]
    vec_c = [0, 0, 1]
    angle_2 = vec_angle(vector_main, vec_c)
    angle_1 = vec_angle([vector_main[0], vector_main[1], 0], vec_a)

    if vector_main[0] >= 0 and vector_main[1] >= 0 and vector_main[2] >= 0:
        angle_1 = -angle_1
        angle_2 = -angle_2
    if vector_main[0] < 0 and vector_main[1] >= 0 and vector_main[2] >= 0:
        angle_1 = np.pi*2 + angle_1
        angle_2 = angle_2
    if vector_main[0] >= 0 and vector_main[1] < 0 and vector_main[2] >= 0:
        angle_1 = angle_1
        angle_2 = -angle_2
    if vector_main[0] < 0 and vector_main[1] < 0 and vector_main[2] >= 0:
        angle_1 = np.pi*2 -angle_1
    if vector_main[0] >= 0 and vector_main[1] >= 0 and vector_main[2] < 0:
        angle_1 = -angle_1
        angle_2 = np.pi + angle_2
    if vector_main[0] < 0 and vector_main[1] >= 0 and vector_main[2] < 0:
        angle_2 = np.pi - angle_2
    if vector_main[0] >= 0 and vector_main[1] < 0 and vector_main[2] < 0:
        angle_2 = angle_2 + np.pi
    if vector_main[0] < 0 and vector_main[1] < 0 and vector_main[2] < 0:
        angle_1 = -angle_1
        angle_2 = np.pi - angle_2

    #First rotation around z-axis with angle_1

    rot_matrix_z = np.array([[np.cos(angle_1), -np.sin(angle_1),      0],
                             [np.sin(angle_1),  np.cos(angle_1),      0],
                             [                0,                  0,      1]])

    #resulting_vector = np.dot(rot_matrix_z, vector_main)

    atom_list_translated = [[i[0],i[1],np.dot(rot_matrix_z, i[2:])[0],
                            np.dot(rot_matrix_z, i[2:])[1],
                            np.dot(rot_matrix_z, i[2:])[2]] for i in atom_list]

    #Second rotation around y-axis with angle_2
    rot_matrix_y = np.array([[ np.cos(angle_2),         0,       np.sin(angle_2)],
                             [               0,          1,                     0],
                             [-np.sin(angle_2),         0,       np.cos(angle_2)]])

    #resulting_vector2 = np.dot(rot_matrix_y, resulting_vector)
    atom_list_translated = [[i[0],i[1],np.dot(rot_matrix_y, i[2:])[0],
                            np.dot(rot_matrix_y, i[2:])[1],
                            np.dot(rot_matrix_y, i[2:])[2]] for i in atom_list_translated]

    #Third step is translation! We are now at approximetely [0,0,-z]
    #We need to shift the origin into the point of the window
    #the value for z we know from the original vector analysis (it is the length on vector where
    #there was the biggest sphere (step - vector_analysed[0]) first value!)
    #We can reason that because you can see that the length of original vector and now the
    #z value for rotated point is the same! the length of vector is preserved
    new_z = vector_analysed[0]
    #old_origin = np.array([0,0,-new_z])
    #new_resulting_vector3 = np.add(resulting_vector2, old_origin)
    atom_list_translated2 = [[i[0],i[1],i[2],i[3],i[4]-new_z] for i in atom_list_translated]

    #!!!Here the xy and z sampling has to take place!!!
    #First sample the same point to check if nothing has changed
    #This point should represent 0,0,0 the origin

    distance_list = []
    for i in atom_list_translated2:
        distance_list.append(np.linalg.norm(np.array([i[2:5]]))-atom_vdw_radii[i[0]])

    x_opt = 0
    y_opt = 0
    z_opt = new_z

    xyz_window = []

    xyz_window.append(2*min([two_points_distance([x_opt,y_opt, z_opt],i[2:5])-atom_vdw_radii[i[0]]                                                    for i in atom_list_translated]))

    parameters1 = (x_opt, y_opt, atom_list_translated)
    rranges_z = ((0.1, 1000),)
    #print(rranges_z)
    normal_optimisation = scipy.optimize.minimize(analyse_z, x0=z_opt, args=parameters1, bounds=rranges_z)
    #brute_optimisation_z = scipy.optimize.brute(analyse_z, rranges_z, args=parameters1, full_output=True)
    #print(brute_optimisation_z)
    #z_opt = brute_optimisation_z[0]
    z_opt = normal_optimisation.x[0]
    #print(z_opt)

    #BRUTE
    rranges = ((-max_value/2, max_value/2), (-max_value/2, max_value/2))
    parameters2 = (z_opt, atom_list_translated)
    brute_optimisation = scipy.optimize.brute(analyse_xy, rranges, args=parameters2,
                                             full_output=True, finish=scipy.optimize.fmin)

    x_opt = brute_optimisation[0][0]
    y_opt = brute_optimisation[0][1]
    xyz_window.append(2*min([two_points_distance([x_opt,y_opt, z_opt],i[2:5])-atom_vdw_radii[i[0]]                                                for i in atom_list_translated]))

    cow = np.array([x_opt, y_opt, z_opt]) #COM of window
    #cow_unaltered = np.array([x_opt, y_opt, z_opt])
    #Reverse translation step
    #rev_resulting_vector3 = np.add(new_resulting_vector3, [0,0,new_z])

    #Reversing the second rotation around axis y
    angle_2_1 = - angle_2
    rev_matrix_y = np.array([[ np.cos(angle_2_1),         0,       np.sin(angle_2_1)],
                             [               0,          1,                     0],
                             [-np.sin(angle_2_1),         0,       np.cos(angle_2_1)]])

    cow = np.dot(rev_matrix_y, cow)
    #rev_resulting_vector4 = np.dot(rev_matrix_y, rev_resulting_vector3)

    #Reversing the first rotation around axis z
    angle_1_1 = - angle_1
    rev_matrix_z = np.array([[np.cos(angle_1_1), -np.sin(angle_1_1),      0],
                             [np.sin(angle_1_1),  np.cos(angle_1_1),      0],
                             [                0,                  0,      1]])

    cow = np.dot(rev_matrix_z, cow)
    #rev_resulting_vector5 = np.dot(rev_matrix_z, rev_resulting_vector4)

    refrence_distance = z_opt / np.linalg.norm(vector_analysed[5:])

    vectors_translated5 = [[np.dot(rot_matrix_z, i[5:])[0],
                           np.dot(rot_matrix_z, i[5:])[1],
                           np.dot(rot_matrix_z, i[5:])[2]] for i in window]

    vectors_translated5 = [[np.dot(rot_matrix_y, i)[0],
                           np.dot(rot_matrix_y, i)[1],
                           np.dot(rot_matrix_y, i)[2]] for i in vectors_translated5]

    cut_points5 = np.array([[i[0]*refrence_distance, i[1]*refrence_distance, i[2]*refrence_distance] for i in vectors_translated5])
    cut_points6 = np.array([[i[0]*refrence_distance, i[1]*refrence_distance] for i in vectors_translated5])

    diameter_points5 = []
    for i in window:
        analysed = vector_analysis(i[5:8], atom_list, atom_coor, atom_vdw, 0.1)
        if analysed != None:
            diameter_points5.append(analysed[1])
        else:
            diameter_points5.append(0.0)

    #for i,j in zip(cut_points5,diameter_points5):
        #print(i,j)

    diameter_points5 = np.array(diameter_points5)

    from matplotlib.mlab import griddata

    xi = np.linspace(min(cut_points5[:,0]), max(cut_points5[:,0]), 100)
    yi = np.linspace(min(cut_points5[:,1]), max(cut_points5[:,1]), 100)
    # grid the data.
    zi = griddata(cut_points5[:,0], cut_points5[:,1], diameter_points5, xi, yi, interp='linear')


    CS = plt.contourf(xi, yi, zi, 10, cmap=plt.cm.rainbow,
                  vmax=abs(diameter_points5).max(), vmin=-abs(diameter_points5).max())

    #CS = plt.contour(xi, yi, zi)

    #plt.clabel(CS, inline=1, fontsize=10)
    plt.colorbar()  # draw colorbar
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(min(cut_points5[:,0])-0.5, max(cut_points5[:,0])+0.5)
    plt.ylim(min(cut_points5[:,1])-0.5, max(cut_points5[:,1])+0.5)

    plt.savefig('output/window_{0}_heat.jpg'.format(window_no))
    plt.show()

    no_of_vectors = len(cut_points5)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=600, facecolor='k')
    #ax.set_axis_bgcolor('k')
    #ax.scatter(cut_points5[:,0], cut_points5[:,1], s=500, lw = 0, color='white')
    #plt.tight_layout()

    plt.scatter(cut_points5[:,0], cut_points5[:,1], s=50, lw = 0, color='white')
    plt.axis('off')

    plt.xlim(min(cut_points5[:,0]) - 1, max(cut_points5[:,0]) + 1)
    plt.ylim(min(cut_points5[:,1]) - 1, max(cut_points5[:,1]) + 1)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('output/window_{0}.jpg'.format(window_no), bbox_inches=extent, facecolor=fig.get_facecolor())
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 12), dpi=600, facecolor='k')

    points = cut_points5[:, [0,1]]
    hull = ConvexHull(points)

    verticesx = np.append(points[hull.vertices,0], points[hull.vertices,0][0]) # to close the convex hull
    verticesy = np.append(points[hull.vertices,1], points[hull.vertices,1][0])

    plt.scatter(points[:,0], points[:,1], s=50, lw = 0, color='white')
    plt.plot(verticesx, verticesy, 'r--', lw=4)
    #plt.fill(verticesx, verticesy, color='w')

    plt.xlim(min(points[:,0]) - 1, max(points[:,0]) + 1)
    plt.ylim(min(points[:,1]) - 1, max(points[:,1]) + 1)

    plt.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('output/window_{0}_convex_overlay.jpg'.format(window_no), bbox_inches=extent, facecolor=fig.get_facecolor())
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 12), dpi=600, facecolor='k')

    points = cut_points5[:, [0,1]]
    hull = ConvexHull(points)

    verticesx = np.append(points[hull.vertices,0], points[hull.vertices,0][0]) # to close the convex hull
    verticesy = np.append(points[hull.vertices,1], points[hull.vertices,1][0])

    plt.plot(verticesx, verticesy, 'w-', lw=4)
    plt.fill(verticesx, verticesy, color='w')

    plt.xlim(min(points[:,0]) - 1, max(points[:,0]) + 1)
    plt.ylim(min(points[:,1]) - 1, max(points[:,1]) + 1)

    plt.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('output/window_{0}_convex_hull.jpg'.format(window_no), bbox_inches=extent, facecolor=fig.get_facecolor())
    plt.show()

    dist_matrix1 = euclidean_distances(cut_points5, cut_points5)
    dist_matrix2 = euclidean_distances(cut_points6, cut_points6)
    final_matrix1 = np.triu(dist_matrix1)
    final_matrix2 = np.triu(dist_matrix2)
    i,j = np.unravel_index(final_matrix1.argmax(), final_matrix1.shape)
    answer1_max = final_matrix1[i,j]
    i,j = np.unravel_index(final_matrix2.argmax(), final_matrix2.shape)
    answer2_max = final_matrix2[i,j]

    #print(answer1_max)
    #print(answer2_max)


    min_1 = []
    for a in final_matrix1:
        m = [i for i in a if i > 0]
        if len(m) > 0:
            min_1.append(min(m))

    db = DBSCAN(eps=0.24).fit(cut_points6)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    sub_output_list = []
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    sub_output_list.append(n_clusters_)

    clusters = []
    for i,j in zip(cut_points6, db.labels_):
        clusters.append([i,j])

    clustered_results = {}
    for i in range(n_clusters_):
        clustered_results[i] = []
        for j,k in zip(clusters,cut_points6):
            if j[1] == i:
                clustered_results[i].append(k)

    #print(len(clustered_results))

    ##############################################################################

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = cut_points6[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = cut_points6[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.show()

    db = DBSCAN(eps=0.24).fit(cut_points5)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_



    sub_output_list = []
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    sub_output_list.append(n_clusters_)

    clusters = []
    for i,j in zip(cut_points6, db.labels_):
        clusters.append([i,j])

    clustered_results = {}
    for i in range(n_clusters_):
        clustered_results[i] = []
        for j,k in zip(clusters,cut_points6):
            if j[1] == i:
                clustered_results[i].append(k)

    #print(len(clustered_results))

    ##############################################################################

    # Black removed and is used for noise instead.

    fig, ax = plt.subplots(figsize=(12, 12), dpi=600, facecolor='k')

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    groups = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy_core = cut_points6[class_member_mask & core_samples_mask]
        plt.plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor='w',
                 markeredgecolor='k', markersize=14)
        groups.append(len(xy_core))

        xy_satelites = cut_points6[class_member_mask & ~core_samples_mask]
        plt.plot(xy_satelites[:, 0], xy_satelites[:, 1], 'o', markerfacecolor='r',
                 markeredgecolor='k', markersize=6)

    plt.xlim(min(cut_points6[:,0]) - 1, max(cut_points6[:,0]) + 1)
    plt.ylim(min(cut_points6[:,1]) - 1, max(cut_points6[:,1]) + 1)
    plt.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('output/window_core_{0}-DBSCAN.jpg'.format(window_no), bbox_inches=extent, facecolor=fig.get_facecolor())
    plt.show()

    #print("\n groups")
    #print(groups)

    index1, value1 = min(enumerate(groups), key=op.itemgetter(1))

    #print(index1, value1)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=600, facecolor='k')

    for k, col in zip(unique_labels, colors):
        #print(k)
        if k == index1-1:
            class_member_mask = (labels == k)
            core = cut_points6[class_member_mask & core_samples_mask]

            plt.scatter(core[:, 0], core[:, 1], s=50, lw = 0, color='white')


    plt.xlim(min(core[:,0]) - 1, max(core[:,0]) + 1)
    plt.ylim(min(core[:,1]) - 1, max(core[:,1]) + 1)

    plt.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('output/window_core_{0}.jpg'.format(window_no), bbox_inches=extent, facecolor=fig.get_facecolor())
    plt.show()



    fig, ax = plt.subplots(figsize=(12, 12), dpi=600, facecolor='k')
    hull = ConvexHull(core)

    verticesx = np.append(core[hull.vertices,0], core[hull.vertices,0][0]) # to close the convex hull
    verticesy = np.append(core[hull.vertices,1], core[hull.vertices,1][0])

    plt.plot(verticesx, verticesy, 'r--', lw=4)
    #plt.fill(verticesx, verticesy, color='w')
    plt.scatter(core[:, 0], core[:, 1], s=50, lw = 0, color='white')
    #plt.plot(array[hull.vertices[0],0], array[hull.vertices[0],1], 'wo')
    plt.xlim(min(core[:,0]) - 1, max(core[:,0]) + 1)
    plt.ylim(min(core[:,1]) - 1, max(core[:,1]) + 1)
    plt.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('output/window_core_{0}_hull_overlay.jpg'.format(window_no), bbox_inches=extent, facecolor=fig.get_facecolor())
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 12), dpi=600, facecolor='k')
    hull = ConvexHull(core)

    verticesx = np.append(core[hull.vertices,0], core[hull.vertices,0][0]) # to close the convex hull
    verticesy = np.append(core[hull.vertices,1], core[hull.vertices,1][0])

    plt.plot(verticesx, verticesy, 'w-', lw=4)
    plt.fill(verticesx, verticesy, color='w')
    #plt.scatter(core[:, 0], core[:, 1], s=50, lw = 0, color='white')
    #plt.plot(array[hull.vertices[0],0], array[hull.vertices[0],1], 'wo')
    plt.xlim(min(core[:,0]) - 1, max(core[:,0]) + 1)
    plt.ylim(min(core[:,1]) - 1, max(core[:,1]) + 1)
    plt.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('output/window_core_{0}_convex_hull.jpg'.format(window_no), bbox_inches=extent, facecolor=fig.get_facecolor())
    plt.show()


    """
    """
    # Compute DBSCAN
    #print('EPS: {}'.format(eps_sqrt))
    #db = DBSCAN(eps=eps_sqrt).fit(cut_points5)
    db = DBSCAN(eps=0.35).fit(cut_points6)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    sub_output_list = []
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    sub_output_list.append(n_clusters_)

    clusters = []
    for i,j in zip(cut_points6, db.labels_):
        clusters.append([i,j])

    clustered_results = {}
    for i in range(n_clusters_):
        clustered_results[i] = []
        for j,k in zip(clusters,cut_points6):
            if j[1] == i:
                clustered_results[i].append(k)

    #print(len(clustered_results))
    #print(clustered_results)

    for i in clustered_results:
        plt.scatter(np.array(clustered_results[i])[:,0], np.array(clustered_results[i])[:,1])
    plt.show()

    ###NEW PART FOR THE SHAPE ANALYSIS
    #We do it after we get the centre of the window as we need to refit the translational matrices to get as most symmetric
    #as possible figure
    refrence_distance = z_opt / np.linalg.norm(vector_analysed[5:])

    vec_a = [1, 0, 0]
    vec_b = [0, 1, 0]
    vec_c = [0, 0, 1]
    angle_2 = vec_angle(cow, vec_c)
    angle_1 = vec_angle([cow[0], cow[1], 0], vec_a)

    if cow[0] >= 0 and cow[1] >= 0 and cow[2] >= 0:
        angle_1 = -angle_1
        angle_2 = -angle_2
    if cow[0] < 0 and cow[1] >= 0 and cow[2] >= 0:
        angle_1 = np.pi*2 + angle_1
        angle_2 = angle_2
    if cow[0] >= 0 and cow[1] < 0 and cow[2] >= 0:
        angle_1 = angle_1
        angle_2 = -angle_2
    if cow[0] < 0 and cow[1] < 0 and cow[2] >= 0:
        angle_1 = np.pi*2 -angle_1
    if cow[0] >= 0 and cow[1] >= 0 and cow[2] < 0:
        angle_1 = -angle_1
        angle_2 = np.pi + angle_2
    if cow[0] < 0 and cow[1] >= 0 and cow[2] < 0:
        angle_2 = np.pi - angle_2
    if cow[0] >= 0 and cow[1] < 0 and cow[2] < 0:
        angle_2 = angle_2 + np.pi
    if cow[0] < 0 and cow[1] < 0 and cow[2] < 0:
        angle_1 = -angle_1
        angle_2 = np.pi - angle_2

    #First rotation around z-axis with angle_1

    rot_matrix_z = np.array([[np.cos(angle_1), -np.sin(angle_1),      0],
                             [np.sin(angle_1),  np.cos(angle_1),      0],
                             [                0,                  0,      1]])

    #Second rotation around y-axis with angle_2
    rot_matrix_y = np.array([[ np.cos(angle_2),         0,       np.sin(angle_2)],
                             [               0,          1,                     0],
                             [-np.sin(angle_2),         0,       np.cos(angle_2)]])


    #This should be the distance where exactly is the centre of the window localised on the z axis parallel to the vector
    #The z axis will be the depth and we will not see it. It will be perpendicular to XY plane we will plot.
    #The shape we want to plot is constructred from the perpendicular vectors and the points that cut the XYZ plane and
    #create some sort of shape

    # 1. We translate all the vectors to be in the same coordinate system as the main vector. We use here the same translation
    # we have used to translate the atom list



    vectors_translated = [[np.dot(rot_matrix_z, i[5:])[0],
                           np.dot(rot_matrix_z, i[5:])[1],
                           np.dot(rot_matrix_z, i[5:])[2]] for i in window]

    vectors_translated = [[np.dot(rot_matrix_y, i)[0],
                           np.dot(rot_matrix_y, i)[1],
                           np.dot(rot_matrix_y, i)[2]] for i in vectors_translated]

    cut_points = np.array([[i[0]*refrence_distance, i[1]*refrence_distance, i[2]*refrence_distance] for i in vectors_translated])
    plt.scatter(cut_points[:,0], cut_points[:,1], lw = 0, color='green')
    plt.show()

    def calculate_correction(vector_a,vector_b):
        #print(vector_a,vector_b)
        #First calculate the angle between these two vectors
        angle_y = vec_angle(vector_a,vector_b)
        #Here the lengths of sides a, b and the angle γ between these sides are known.
        #The third side can be determined from the law of cosines sqrt(a^2+b^2-2*a*b*cosy) also a=b
        a = b = np.linalg.norm(vector_a)
        c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(angle_y))
        #print('\nA: {0}, B: {1}, Gamma: {2}'.format(np.linalg.norm(vector_a), np.linalg.norm(vector_a), angle_y))
        #print('Angle gamma {0}'.format(np.rad2deg(angle_y)))
        #print('Answer c length: {0}'.format(c))

        angle_b = angle_a = np.arccos((b**2+c**2-a**2)/(2*b*c))

        #print('Angle beta {0}'.format(np.rad2deg(angle_b)))
        #print('Angle alfa {0}\n'.format(np.rad2deg(angle_a)))

        #Now calculate the correction length
        angle_b_prim = 0.5 * np.pi - angle_b
        angle_a_prim = np.pi - angle_a
        angle_y_prim = np.pi - angle_a_prim - angle_b_prim

        #print('New alpha {} beta {} gamma {}'.format(angle_a_prim, angle_b_prim, angle_y_prim))

        c_prim = c

        a_prim = c_prim * (np.sin(angle_b_prim)/np.sin(angle_y_prim))
        b_prim = c_prim * (np.sin(angle_a_prim)/np.sin(angle_y_prim))

        #print('New a: {} b: {} c: {}'.format(a_prim, b_prim, c_prim))
        return(a_prim)

    cut_points1 = []
    cut_points2 = []
    for i in vectors_translated:
        corr = calculate_correction(cow_unaltered, i) + refrence_distance
        cut_points1.append([i[0]*corr, i[1]*corr, i[2]*corr])
        cut_points2.append([i[0]*corr, i[1]*corr])
    cut_points1 = np.array(cut_points1)
    cut_points2 = np.array(cut_points2)

    plt.scatter(cut_points1[:,0], cut_points1[:,1], lw = 0, color='red')
    plt.show()

    # Compute DBSCAN
    print('EPS: {}'.format(eps_sqrt))
    #db = DBSCAN(eps=eps_sqrt).fit(cut_points5)
    db = DBSCAN(eps=2).fit(cut_points1)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    sub_output_list = []
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    sub_output_list.append(n_clusters_)

    clusters = []
    for i,j in zip(cut_points1, db.labels_):
        clusters.append([i,j])

    clustered_results = {}
    for i in range(n_clusters_):
        clustered_results[i] = []
        for j,k in zip(clusters,cut_points1):
            if j[1] == i:
                clustered_results[i].append(k)

    print(len(clustered_results))
    #print(clustered_results)

    for i in clustered_results:
        plt.scatter(np.array(clustered_results[i])[:,0], np.array(clustered_results[i])[:,1])
    plt.show()

    return([xyz_window[1], cow])


# In[ ]:
"""
