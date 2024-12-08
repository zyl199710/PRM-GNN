import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def num_node(file1,file2):
    d_label = dict()
    d = dict()
    d_power = dict()
    node = []
    num = 0
    for lines in file1.readlines():
        line1 = lines.split('\t')
        d_power[int(line1[0])] = int(line1[1])
    for lines in file2.readlines():
        line1 = lines.split('\n')
        line = line1[0].split('\t')
        tuple = int(line[0]), int(line[1])
        if tuple[0] > tuple[1]:
            tuple = tuple[1], tuple[0]
        if d_power[tuple[0]] < d_power[tuple[1]]:
            tuple = tuple[1] , tuple[0]
        if tuple not in node:
            d[tuple] = num
            node.append(tuple)  # 作者-
            if d_power[tuple[0]] > d_power[tuple[1]]:
                d_label[d[tuple]] = 1
            else:
                d_label[d[tuple]] = 0
            num = num + 1
    file2.close()
    G = nx.Graph()
    G.add_edges_from(node)
    print(num)
    return G, d, d_label


def generate_W(d):
    G = nx.Graph()
    edge = []
    edges = []
    tuple0 = []
    file = open('dataset/data-enron/edge.txt', "w")
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
                    file.write(str(edge[0])+'\t'+str(edge[1])+'\n')
    file.close
    np.save("dataset/data-enron/edge-enron.npy", edges)


def label_list(d, d_label):
    #    file1 = open('dataset/Coauthor/mini')


    label = []
    l1,l2 = 0,0
    for i in d_label:
        if d_label[i] == 1:
            l1 = l1 +1
        else:
            l2 = l2 +1

        label.append(d_label[i])
    print(l1,l2)
    np.save("dataset/data-enron/label-enron.npy", label)
    print(len(label))
   # file = open('dataset/data-enron/label.txt',"w")
   # for l in label:
   #     file.write(str(l)+'\n')
   # file.close
def getemail(file2):
    d_sen = dict()
    d_rec = dict()
    for lines in file2.readlines():
        line1 = lines.split('\n')
        line = line1[0].split('\t')
        i, j = int(line[0]), int(line[1])
        tuple = int(line[0]), int(line[1])
        if i not in d_sen:
            d_sen[i] = []
        if j not in d_rec:
            d_rec[j] = []
        d_sen[i].append(j)
        d_rec[j].append(i)



    return d_sen, d_rec
def getfeature(d_sen, d_rec, d, n_pr):
    feature = []
    features = []
    for tuple in d:
        i = int(tuple[0])
        j = int(tuple[1])
        sen1 = 0
        sen2 = 0
        sen3 = 0
        sen4 = 0
        rec1 = 0
        rec2 = 0
        rec3 = 0
        rec4 = 0
        if i not in d_sen:
            sen1 = 0
            sen2 = 0
        else:
            for k in d_sen[i]:
                if k == j:
                    sen1 = sen1 + 1
                else:
                    sen2 = sen2 + 1
        if j not in d_sen:
            sen3 = 0
            sen4 = 0
        else:
            for k in d_sen[j]:
                if k == i:
                    sen3 = sen1 + 1
                else:
                    sen4 = sen2 + 1

        if j not in d_rec:
            rec1 = 0
            rec2 = 0
        else:
            for l in d_rec[j]:
                if l == i :
                    rec1 = rec1 + 1
                else:
                    rec2 = rec2 + 1

        if i not in d_rec:
            rec3 = 0
            rec4 = 0
        else:
            for l in d_rec[i]:
                if l == j:
                    rec3 = rec3 + 1
                else:
                    rec4= rec4 + 1
        feature.append(sen1 + sen2)
        feature.append(sen3 + sen4 )
        feature.append(n_pr[i])
        feature.append(n_pr[j])
        feature.append(sen1)
        feature.append(sen3)
        feature.append(sen2)
        feature.append(sen4)
        feature.append(rec2)
        feature.append(rec4)
        feature.append(n_pr[i]- n_pr[j])
        features.append(feature)
        feature = []
    np.save("dataset/data-enron/feature-enron.npy", features)
    print(len(features))


# np.save("feature.npy",features)


if __name__ == '__main__':
   # file1 = open('dataset/Coauthor/coauthor.txt')
    #file = open('dataset/Coauthor/coauthor.txt')
    file1 = open('dataset/enron/enron_list.txt')
    file2 = open('dataset/enron/enron_edge.txt')
    file3 = open('dataset/enron/enron_edge.txt')
    #n_pr = pagerank(file1)
    G, d, d_label = num_node(file1,file2)
    d_sen, d_rec = getemail(file3)
    n_pr = nx.pagerank(G)
    getfeature(d_sen, d_rec, d, n_pr)

   # print('dict finished')
    generate_W(d)
    #print('W finished')
    label_list(d,d_label)
