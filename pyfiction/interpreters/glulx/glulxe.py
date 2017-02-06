import os

from pyfiction import PYFICTION_PATH
from pyfiction.interpreters.interpreter import Interpreter


class Glulxe(Interpreter):
    name = 'glulxe'
    description = 'cheapglulxe - cheapglk + glulxe compiled interpreter.'
    path = os.path.join(PYFICTION_PATH, 'interpreters/glulx/cheapglulxe')
