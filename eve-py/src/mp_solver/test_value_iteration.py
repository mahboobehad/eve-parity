import unittest

from igraph import Graph

from .value_iteration import solve_energy_game, solve_mp_game


class TestValueIterationAlgorithm(unittest.TestCase):

    def test_solve_energy_game_finds_expected_values(self):
        g = Graph(directed=True)

        x = g.add_vertex(label="x", player=1)
        v = g.add_vertex(label="v", player=1)
        y = g.add_vertex(label="y", player=0)
        z = g.add_vertex(label="z", player=0)
        w = g.add_vertex(label="w", player=0)

        g.add_edge(source=x, target=z, weight=3)
        g.add_edge(source=y, target=x, weight=2)
        g.add_edge(source=y, target=z, weight=1)
        g.add_edge(source=z, target=y, weight=-3)
        g.add_edge(source=z, target=w, weight=1)
        g.add_edge(source=v, target=z, weight=1)
        g.add_edge(source=v, target=w, weight=0)
        g.add_edge(source=w, target=v, weight=-4)

        values = solve_energy_game(g)

        self.assertEqual(values['x'], 0)
        self.assertEqual(values['z'], 3)
        self.assertEqual(values['y'], 0)


class TestMeanPayoffSolver(unittest.TestCase):

    def test_solve_mp_game_finds_expected_values(self):
        g = Graph(directed=True)

        x = g.add_vertex(label="x", player=1)
        v = g.add_vertex(label="v", player=1)
        y = g.add_vertex(label="y", player=0)
        z = g.add_vertex(label="z", player=0)
        w = g.add_vertex(label="w", player=0)

        g.add_edge(source=x, target=z, weight=3)
        g.add_edge(source=y, target=x, weight=2)
        g.add_edge(source=y, target=z, weight=1)
        g.add_edge(source=z, target=y, weight=-3)
        g.add_edge(source=z, target=w, weight=1)
        g.add_edge(source=v, target=z, weight=1)
        g.add_edge(source=v, target=w, weight=0)
        g.add_edge(source=w, target=v, weight=-4)

        values = solve_mp_game(g)

        self.assertEqual(len(values), 5, "should have values for all 5 vertices")

        self.assertLess(values['v'], 0)
        self.assertLess(values['w'], 0)

        self.assertGreater(values['x'], 0)
        self.assertGreater(values['y'], 0)
        self.assertGreater(values['z'], 0)

        v_val = values['v']
        w_val = values['w']
        x_val = values['x']
        y_val = values['y']
        z_val = values['z']

        print(f"v: {v_val}, w: {w_val}, x: {x_val}, y: {y_val}, z: {z_val}")

        self.assertAlmostEqual(v_val, w_val, places=1)

        self.assertAlmostEqual(x_val, y_val, places=1)
        self.assertAlmostEqual(y_val, z_val, places=1)

        self.assertLess(v_val, x_val)
        self.assertLess(w_val, y_val)


if __name__ == '__main__':
    unittest.main()
