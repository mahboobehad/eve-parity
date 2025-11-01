import unittest

from igraph import Graph

from .value_iteration import solve_energy_game


class TestValueIterationAlgorithm(unittest.TestCase):

    def test_finds_expected_values(self):
        g = Graph(directed=True)

        x = g.add_vertex(label="x", player="1")
        v = g.add_vertex(label="v", player="1")
        y = g.add_vertex(label="y", player="0")
        z = g.add_vertex(label="z", player="0")
        w = g.add_vertex(label="w", player="0")

        g.add_edge(source=x, target=z, weight=3)
        g.add_edge(source=y, target=x, weight=2)
        g.add_edge(source=y, target=z, weight=1)
        g.add_edge(source=z, target=y, weight=-3)
        g.add_edge(source=z, target=w, weight=1)
        g.add_edge(source=v, target=z, weight=1)
        g.add_edge(source=v, target=w, weight=0)
        g.add_edge(source=w, target=v, weight=-4)

        solve_energy_game(g)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
