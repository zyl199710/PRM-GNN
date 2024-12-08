import itertools
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def author_paper(file):
    tuple = []
    G = nx.Graph()
    d_a_a_p = {}
    edges = []
    d_a_p = dict()
    d_p_a = dict()
    for lines in file.readlines():
        line = lines.split('\t')
        if int(line[0]) not in d_p_a:
            d_p_a[int(line[0])] = []
        if int(line[1]) not in d_p_a[int(line[0])]:
            d_p_a[int(line[0])].append(int(line[1]))

        if int(line[1]) not in d_a_p:
            d_a_p[int(line[1])] = []
        if int(line[0]) not in d_a_p[int(line[1])]:
            d_a_p[int(line[1])].append(int(line[0]))
    #file3 = open('dataset\Coauthor\ author_paper_mini.txt',"w")

    for i in d_a_p:
        for j in d_a_p[i]:
            tuple = i,j
            edges.append(tuple)
            #file3.write(i+'\t'+j+'\n')

    file = open('dataset/Coauthor/paper_year.txt')
    d_p_y = dict()
    d_a_y = dict()
    for lines in file.readlines():
        line1 = lines.split('\n')
        line = line1[0].split('\t')
        d_p_y[int(line[0])]= int(line[1])
    for i in d_a_p:
        l = []
        for j in d_a_p[i]:
            l.append(d_p_y[j])
        l.sort()
        d_a_y[i] = l[0]

    for i in d_p_a:
        for j1 in d_p_a[i]:
            for j2 in d_p_a[i]:
                i1, i2 = min(j1, j2), max(j1, j2)
                if d_a_y[i1] < d_a_y[i2]:
                    i1, i2 = i2, i1
                tuple = i1, i2
                if tuple not in d_a_a_p:
                    d_a_a_p[tuple] = []
                if i not in d_a_a_p[tuple]:
                    d_a_a_p[tuple].append(i)
    #print(d_a_a_p)
    return d_a_p,d_p_a,d_a_a_p,d_a_y
''''''



def number_node(file1,file2,file3,file4,file5,d_p_a,d_a_p):
    num = 1
    d = dict()
    d_label = dict()
    edges = []
    nodes = []
    G = nx.Graph()
    d_n_adj=dict()
    '''
    for lines in file1.readlines():
        lines = lines.strip('\n')
        line = lines.split('\t')
        tuple = int(line[0]), int(line[1])

        tuple = min(tuple[0], tuple[1]), max(tuple[0], tuple[1])
        d_label[tuple] = 0
        if tuple not in edges:
            d[tuple] = num
            edges.append(tuple)  # 作者-
            num = num + 1
'''
    for lines in file2.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]),int(line[1])

        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 = tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    for lines in file3.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]), int(line[1])

        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 = tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    for lines in file4.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]), int(line[1])


        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 = tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    for lines in file5.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]), int(line[1])

        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 =tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    G.add_edges_from(edges)
    sub_G = max(nx.connected_components(G), key=len)
    small_components = sorted(nx.connected_components(G), key=len)[:-1]  #前三极大图
    G.remove_nodes_from(itertools.chain.from_iterable(small_components))
    print(len(G.edges))
    for p in d_p_a:
        l = set(d_p_a[p]).intersection(G.nodes)
        #print(l)
        if len(l) >= 1:
            for i in range(len(d_p_a[p])):
                for j in range(i+1,len(d_p_a[p])):
                    i1,i2 = min(int(d_p_a[p][i]),int(d_p_a[p][j])), max(int(d_p_a[p][i]),int(d_p_a[p][j]))
                    tuple = i1 , i2
                    #print(tuple)
                    if tuple not in edges:
                        edges.append(tuple)
                        d[tuple] = num
                        i1, i2 = tuple
                        tuple1 = i2, i1
                        edges.append(tuple1)
                        num = num+1
                        d_label[tuple] = 0

    #print(d,len(d))
    G.add_edges_from(edges)
  #  print(len(d))
   # small_components = sorted(nx.connected_components(G), key=len)[:-1]  # 前三极大图
   # G.remove_nodes_from(itertools.chain.from_iterable(small_components))
    A = np.array(nx.adjacency_matrix(G).todense())
    i = 0
    for node in G.nodes:
        d_n_adj[node] = A[i]
        i = i+1
    print(d_n_adj)

   # print(A, A[0])
 #   print(len(G.edges),len(G.nodes))
  #  print(num)

    '''
    nx.draw_networkx(G, node_size=50, with_labels = False)  # pos=nx.spring_layout(G)
    plt.axis('off')
    plt.show()
'''
#   print(d)
#   print(d_label)
    return G,d_label,d, d_n_adj

def getlabel(d,d_label):
    l1,l2,l3=0,0,0
    label = []
    for tuple in d:
        if tuple not in d_label:
            d_label[tuple] = 0
        label.append(d_label[tuple])
    np.save("label.npy",label)
def getnode():
    nodes = []
    for tuple in d:
        if tuple[0] not in nodes:
           nodes.append(tuple[0])
        if tuple[1] not in nodes:
           nodes.append(tuple[1])
    print(len(nodes))
    print(len(d))

def getedge(d):
    edges = []
    G1 = nx.Graph()
    for tuple1 in d:
        for tuple2 in d:
            l = set(tuple1).intersection(tuple2)
            if len(l)==1:
                i1 = d[tuple1]
                i2 = d[tuple2]
                if i1>i2:
                    i1,i2 = i2,i1

                edge = i1,i2
                if edge not in edges:
                    edges.append(edge)

  #  np.save("edge.npy",edges)
    G1.add_edges_from(edges)
    print(len(G1.nodes))


    '''
def sample_with_RWR_and_three_degree_infuence(K_node,G):
    #重启随机游走算法（RWR）
    # 基于重启随机游走一次采样
    N = 5000
    re_p = 0.2
    RWR_node = [K_node]
    switch_node = K_node
    while (True):
        p = float(1 - re_p) / len(list(G.neighbors(switch_node)))
        choice_list = list(G.neighbors(switch_node))
        choice_list.append(K_node)
        choice_probability = [p] * len(list(G.neighbors(switch_node)))
        choice_probability.append(re_p)
        # 随机数
        random_number = random.uniform(0, 1)
        for i in range(0, len(choice_list)):
            if random_number < sum(choice_probability[0:i + 1]):
                switch_node = choice_list[i]
                break
        if switch_node not in RWR_node:
            RWR_node.append(switch_node)
        if len(RWR_node) >= N:
            break
        # 基于三度影响力二次采样
    sub_G = nx.Graph()
    edges = []
    for node_1 in G.neighbors(K_node):
        if node_1 in RWR_node:
            item = K_node, node_1
            edges.append(item)
        for node_2 in G.neighbors(node_1):
            if node_2 in RWR_node:
                item = node_1, node_2
                edges.append(item)
            for node_3 in G.neighbors(node_2):
                if node_3 in RWR_node:
                    item = node_2, node_3
                    edges.append(item)
    edges = list(set(edges))
    sub_G.add_edges_from(edges)
    return sub_G
'''
def a_firstyear(d_a_p):
    file = open('dataset/Coauthor/paper_year.txt')
    d_p_y = dict()
    d_a_y = dict()
    for lines in file.readlines():
        line1 = lines.split('\n')
        line = line1[0].split('\t')
        d_p_y[int(line[0])]= int(line[1])
    for i in d_a_p:
        l = []
        for j in d_a_p[i]:
            l.append(d_p_y[j])
        l.sort()
        d_a_y[i] = l[0]
    return d_a_y

def getfeature(d_a_p,d,d_a_a_p,d_a_y,n_pr):
    feature = []
    features = []
    for tuple in d:
        i = tuple[0]
        j = tuple[1]
        feature.append(len(d_a_p[i]))
        feature.append(len(d_a_p[j]))
        feature.append(len(d_a_p[i]) / len(d_a_p[j]))
        l1 = d_a_p[tuple[0]]
        l2 = d_a_p[tuple[1]]
        p_cp = len(set(l1).intersection(l2))
        feature.append(p_cp / len(d_a_p[i]))
        feature.append(p_cp / len(d_a_p[j]))
        feature.append(d_a_y[i] - d_a_y[j])
        feature.append(n_pr[i] / n_pr[j] )

        features.append(feature)
        feature = []
    print(len(features))
   # np.save("feature.npy",features)
def getshifufeature(d_a_p,d,d_a_a_p,d_a_y,n_pr, d_n_adj):
    feature = []
    features = []
    for tuple in d:
        i = tuple[0]
        j = tuple[1]
        feature.append(len(d_a_p[i]))
        feature.append(len(d_a_p[j]))
        feature.append(len(d_a_p[i]) / len(d_a_p[j]))
        l1 = d_a_p[tuple[0]]
        l2 = d_a_p[tuple[1]]
        p_cp = len(set(l1).intersection(l2))
        feature.append(p_cp / len(d_a_p[i]))
        feature.append(p_cp / len(d_a_p[j]))
        feature.append(d_a_y[i] - d_a_y[j])
        feature.append(n_pr[i] / n_pr[j] )

        feature.extend(d_n_adj[i])
        feature.extend(d_n_adj[j])
        print(feature,len(feature))
        features.append(feature)

        feature = []
    print(len(features))
    np.save("feature-shifu.npy",features)
if __name__ == '__main__':
  # file1 = open('dataset/Coauthor/coauthor.txt')
   #file = open('dataset/Coauthor/coauthor.txt')
    file1 = open('dataset/Coauthor/colleague.txt')
    file2 = open('dataset/Coauthor/phd.ans')
    file3 = open('dataset/Coauthor/MathGenealogy_50896.ans')
    file4 = open('dataset/Coauthor/teacher.ans')
    file5 = open('dataset/Coauthor/ai.ans')
    file6 = open('dataset/Coauthor/paper_author.txt')
    #file7 = open('pred-crf.txt')
    #file8 = open('')
    d_a_p, d_p_a, d_a_a_p,d_a_y= author_paper(file6)

    G,d_label,d, d_n_adj= number_node(file1,file2,file3,file4,file5,d_p_a,d_a_p)
  #  getnode()
 #   pos = nx.spring_layout(G)
 #   nx.draw(G, pos, with_labels=False)
  #  plt.show()
   # sub_G = max(nx.connected_components(G),key = len)
   # small_components = sorted(nx.connected_components(G), key=len)[:-1]  #前三极大图
   # G.remove_nodes_from(itertools.chain.from_iterable(small_components))

   # getedge(d)
 #   getlabel(d, d_label)
    n_pr = nx.pagerank(G)

  #  getfeature(d_a_p,d,d_a_a_p,d_a_y,n_pr,A)
    getshifufeature(d_a_p, d, d_a_a_p, d_a_y, n_pr, d_n_adj)

  #  l1,l2=0,0
  #  for tuple in d:
   #     if d_label[tuple] == 1:
   #         l1=l1+1
   #     else:
    #        l2 = l2+1
    #print(l1,l2)
   # print('dict finished')

    #
    #d_f = friends(d_p_a)
    #generate_W(d)
    #print(d,len(d))
   # label = label_list(d,d_label)
