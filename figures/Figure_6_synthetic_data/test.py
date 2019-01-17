def match_cells(gt_cells, m_cells, storm_input, filtered_binaries, max_d=3):
    img_numbers = np.array([int(re.findall(r'(\d+)', cell.name)[0]) for cell in m_cells])
    encoded_gt = encode_intensity(gt_cells[:9000])

    d_ = []

    gt_matched, m_matched = [], []
    for i in tqdm(np.unique(storm_input['frame'])[:5]):  # Iteration starts at 1 (ImageJ indexing)
        st_elem = storm_input[storm_input['frame'] == i].copy()
        X = np.array([st_elem['x'], st_elem['y']]).T.copy()
        linkage = fc.linkage(X)
        clusters = fcluster(linkage, max_d, criterion='distance')
        clustered_st = [st_elem[clusters == i] for i in np.unique(clusters)]
        encoded_storm = [encode(elem['intensity']) for elem in clustered_st]

        s_cells = m_cells[img_numbers == (i - 1)]
        if len(s_cells) == 0:
            print('No cells, img {}'.format(i))
            continue

        cell_numbers = np.array([int(re.findall(r'(\d+)', cell.name)[1]) for cell in s_cells])
        binary_img = filtered_binaries[i - 1]
        coms_cells = np.array([mh.center_of_mass(binary_img == j) for j in cell_numbers])

        bordering = 0
        too_far = 0
        for cluster, code in zip(clustered_storm, encoded_storm):

            # Find the GT cell
            idx_gt = np.argwhere(code == encoded_gt)
            if len(idx_gt) == 0:
                # print('Cluster not in cells, probably bordering cell')
                bordering += 1
                continue
            else:
                gt_cell = gt_cells[idx_gt[0][0]]

            print('idx', i)
            print(coms_cells)

            # Find the M cell
            com_storm = [np.mean(cluster['y']), np.mean(cluster['x'])]
            print('com_storm', com_storm)
            ds = np.sqrt((coms_cells[:, 0] - com_storm[0]) ** 2 + (coms_cells[:, 1] - com_storm[1]) ** 2)
            print('ds', ds)

            idx_m = np.argmin(ds)
            if np.min(ds) > 10:
                print('too far')
                too_far += 1
                d_.append(np.min(ds))
                continue
            else:
                m_cell = s_cells[idx_m]

            gt_matched.append(gt_cell)
            m_matched.append(m_cell)
        if not bordering == too_far:
            print(bordering, too_far)
            print('Failure in index {}'.format(i))

        plt.figure()
        plt.imshow(binary_img)
        for c in clustered_st:
            plt.scatter(c['x'], c['y'])

    return gt_matched, m_matched, d_