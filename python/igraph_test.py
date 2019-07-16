#coding = 'utf-8'
import jgraph

g = jgraph.Graph()
vertex = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
g.add_vertices(vertex)

edges = [('a', 'c'), ('a', 'e'), ('a', 'b'), ('b', 'd'), ('b', 'g'), ('c', 'e'),
         ('d', 'f'), ('d', 'g'), ('e', 'f'), ('e', 'g'), ('f', 'g')]
g.add_edges(edges)

g.vs['label'] =  ['A', 'B', 'C', 'D', 'E', 'F', 'G']

g.vs['aera'] = [50, 100, 70, 40, 60, 40, 80]

g['Date'] = '279'

layout = g.layout('kk')

jgraph.plot(g, layout)