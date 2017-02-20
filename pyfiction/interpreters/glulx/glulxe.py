import os

from pyfiction.interpreters.interpreter import Interpreter


class Glulxe(Interpreter):
    name = 'glulxe'
    description = 'cheapglulxe - cheapglk + glulxe compiled interpreter.'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cheapglulxe')
