import os


def read_graph(cwd, directory, output):
    files = os.listdir(cwd + directory)
    files = [x for x in files if x[-5:] == 'edges']

    nodes = set()
    edges_raw = []
    for file_name in files:
        # print('file_name', file_name)
        with open(cwd + directory + file_name, 'r') as f:
            for l in f:
                node_from, node_to = l.split()
                node_from = int(node_from)
                node_to = int(node_to)
                #
                edges_raw.append((node_from, node_to))
                nodes.add(node_from)
                nodes.add(node_to)

    nodes_map = {}
    i = 0
    for n in sorted(nodes):
        nodes_map[n] = i
        i = i + 1
    max = i-1
    
    # graph = []
    graph = {}
    for i in range(max+1):
        graph[i] = set()

    for tup in edges_raw:
        n_from, n_to = tup
        n_from = nodes_map[n_from]
        n_to = nodes_map[n_to]
        graph[n_to].add(n_from)


    edges_range = []
    edges = []
    i = 0
    edges_range.append(i)
    for n in sorted(graph):
        e = sorted(graph[n])
        for nn in e:
            edges.append(nn)
            i += 1
        edges_range.append(i)
    edges_range.append(len(edges))

    with open(cwd + output + '.nodes', 'w') as f:
        f.write(str(len(edges_range)) + '\n')
        for i in edges_range:
            f.write(str(i) + '\n')
    with open(cwd + output + '.edges', 'w') as f:
        f.write(str(len(edges)) + '\n')
        for i in edges:
            f.write(str(i) + '\n')

    print('done!')
    return True


read_graph('./dataset/', 'twitter/', 'twitter-all')