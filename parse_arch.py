"""Parse architecture description YAML file, output machine-readable list"""
# TODO: It seems obviously better to send a number of instances to the API
#  rather than just calling the API a number of times with the exact same
#  element. But does it make sense for every element. Or just neurons?
import yaml

MAX_RECURSION = 32

def parse(arch_dict):
    print(arch_dict)

    # On the first pass, parse anything that isn't "sim"
    #  The user can define custom blocks that can be reused
    #  A globally defined block can't contain other structures? lets see

    if "architecture" not in arch_dict:
        print("Error: no architecture defined")
    else:
        parse_struct(arch_dict["architecture"], [], 1)

    return


def parse_range(range_str): 
    range_str = range_str.replace("]", "")
    range_str = range_str.split("[")[1]
    range_min = int(range_str.split("..")[0])
    range_max = int(range_str.split("..")[1])

    return range_min, range_max


def parse_struct(struct_dict, parent_elements, recursion_depth):
    if recursion_depth > MAX_RECURSION:
        raise Exception("Error: Exceeded max recursion depth ({0}), "
                        "stopping!".format(MAX_RECURSION))

    struct_name = struct_dict["name"]
    # Work out how many instances of this structure to create
    if "[" in struct_name:
        # Can use notation [min..max] to indicate range of elements
        range_min, range_max = parse_range(struct_name)
    else:
        range_min, range_max = 0, 0

    struct_elements = list()
    for instance in range(range_min, range_max+1):
        struct_name = struct_name.split("[")[0] + "[{0}]".format(instance)
        print("Parsing struct {0}".format(struct_name))
 
        # Add any elements local to this h/w structure. They have access to any
        #  elements in the parent structures
        if "local" in struct_dict:
            local_elements = parse_local(struct_dict["local"], parent_elements)
        else:
            local_elements = dict()
        
        # When looking at the subtree, elements may use information about this
        #  structure and any parent structures 
        subtree_elements = list()
        elements = combine_elements(local_elements, parent_elements)
        if "subtree" in struct_dict:
            parse_subtree(struct_dict["subtree"], elements, recursion_depth+1)

    return


def combine_elements(local, parent):
    print(local)
    combined = dict(local)
    for key in parent:
        if key not in combined:
            combined[key] = []

        combined[key].append(parent[key])
    return combined


def parse_subtree(subtree, parent_elements, recursion_depth):
    print("Parsing subtree, parent: {0}".format(parent_elements))
    if not isinstance(subtree, list):
        raise Exception("Subtree must have list of branches")

    for branch in subtree:
        parse_struct(branch, parent_elements, recursion_depth)


def parse_local(local, parent_ids):
    if not isinstance(local, list):
        raise Exception("Local vars must be list of elements")

    # Now create all the local elements in the simulation or arch description
    # Create all non-neuron elements first, then pass these as a reference
    #  to the neuron 
    elements = {"memory": [], "synapse": [], "neuron": [],
                "dendrite": [], "axon_in": [], "axon_out": [], "router": [],}
    for el in local:
        # Group elements of the same class together   
        # TODO: rename to get_instances ? 
        el["instances"] = get_instances(el)
        class_name = el["class"]
        elements[class_name].append(el)

    routers = [parse_router(el) for el in elements["router"]]
    memories = [parse_memory(el) for el in elements["memory"]]
    dendrites = [parse_dendrite(el) for el in elements["dendrite"]]
    synapses = [parse_synapse(el) for el in elements["synapse"]]

    local_ids = { "routers": routers, "memories": memories,
                  "dendrites": dendrites, "synapses": synapses }

    # These elements need to know about other elements that are local or higher
    #  in the hierarchy
    if "routers" in parent_ids:
        routers = routers + parent_ids["routers"]
    local_ids["axon_inputs"] = []
    for el in elements["axon_in"]:
        el["routers"] = routers
        local_ids["axon_inputs"].append(parse_axon_in(el))

    local_ids["axon_outputs"] = []
    for el in elements["axon_out"]:
        el["routers"] = routers
        local_ids["axon_outputs"].append(parse_axon_out(el))

    # Need to combine both parent and local elements and pass these to the
    #  neuron
    dependencies = combine_elements(local_ids, parent_ids)
    neurons = []
    for el in elements["neuron"]:
        el["synapses"] = dependencies["synapses"]
        el["dendrites"] = dependencies["dendrites"]
        el["synapses"] = dependencies["synapses"]
        el["axon_inputs"] = dependencies["axon_inputs"]
        el["axon_outputs"] = dependencies["axon_outputs"]

        timer = create_timer()
        local_ids["timer"] = timer
        el["timer"] = timer

        neuron = parse_neuron(el)

    return local_ids


def get_instances(element_dict):
    element_name = element_dict["name"]

    if "[" in element_name:
        range_min, range_max = parse_range(element_name)
        instances = (range_max - range_min) + 1
    else:
        instances = 1

    print("Parsing element {0}".format(element_name)) 
    return instances


def parse_neuron(element_dict):
    global _neuron_count

    instances = element_dict["instances"]
    attributes = element_dict["attributes"]

    neuron_ids = create_neuron(instances, element_dict["synapses"],
                                          element_dict["dendrites"],
                                          element_dict["axon_inputs"],
                                          element_dict["axon_outputs"],
                                          element_dict["timer"])
    return [neuron_ids]


def parse_router(element_dict):
    router_ids = []

    assert(element_dict["instances"] == 1)

    attributes = element_dict["attributes"]
    if "connection" in attributes:
        connection_type = attributes["connection"]
        dimensions = int(attributes["dimensions"])
        width = int(attributes["width"])

    return create_router(dimensions, width, connection_type)


def parse_synapse(element_dict):
    attributes = element_dict["attributes"]

    print(attributes)
    model = attributes["model"]
    weight_bits = attributes["weight_bits"]

    return create_synapse(model, weight_bits)


def parse_axon_out(element_dict):
    routers = element_dict["routers"]
    assert(len(routers) == 1)
    return create_axon_out(routers[0])


def parse_axon_in(element_dict):
    routers = element_dict["routers"]
    assert(len(routers) == 1)
    return create_axon_in(routers[0])


def parse_memory(element_dict):
    attributes = element_dict["attributes"]
    mem_size = attributes["size"]
    return create_memory(mem_size)
"""
Functions to add elements. These can be swapped out for python API calls in
the future.
"""
_neurons = []
_neuron_count = 0
def create_neuron(instances, synapses, dendrites, axon_inputs, axon_outputs,
                                                                        timer):
    global _neuron_count

    neuron_id = _neuron_count
    _neuron_count += instances
    if instances > 1:
        id_str= "{0}..{1}".format(neuron_id,
                                  (neuron_id+instances) - 1)
    else:
        id_str = str(neuron_id)

    neuron = "n {0} {1} {2} {3} {4}".format(id_str, synapses[0],
                                            axon_inputs[0], axon_outputs[0],
                                            timer)
    _neurons.append(neuron)

    # TODO: return list of ids
    return neuron_id


_routers = []
def create_router(dimensions, width, connection_type):
    router_id = len(_routers)

    if connection_type == "mesh":
        assert(dimensions == 2)
        x = router_id % width
        y = int(router_id / width)

    router = "r {0} {1} {2}".format(router_id, x, y) 
    _routers.append(router)
    return router_id


_synapses = []
def create_synapse(synapse_model, bits):
    synapse_id = len(_synapses)

    models = { "cuba": 0 }
    model_id = models[synapse_model]

    synapse = "s {0} {1} {2}".format(synapse_id, model_id, bits)
    _synapses.append(synapse)

    return synapse_id


_dendrites = []
def parse_dendrite(element_dict):
    return create_dendrite()


def create_dendrite():
    dendrite_id = len(_dendrites)
    dendrite = "d {0}".format(dendrite_id)
    _dendrites.append(dendrite)

    return dendrite_id


_memories = []
def create_memory(mem_size):
    memory_id = len(_memories)
    memory = "m {0} {1}".format(memory_id, mem_size, )
    _memories.append(memory)

    return memory_id


_axon_inputs = []
def create_axon_in(router_id):
    axon_id = len(_axon_inputs)
    axon = "i {0} {1}".format(axon_id, router_id)
    _axon_inputs.append(axon)

    return axon_id


_axon_outputs = []
def create_axon_out(router_id):
    axon_id = len(_axon_outputs)
    axon = "o {0} {1}".format(axon_id, router_id)
    _axon_outputs.append(axon)

    return axon_id


_timers = []
def create_timer():
    timer_id = len(_timers)
    timer = "t {0}".format(timer_id)
    _timers.append(timer)

    return timer_id


if __name__ == "__main__":
    arch_list = None
    with open("loihi.yaml", "r") as arch_file:
        arch_dict = yaml.safe_load(arch_file)
        parse(arch_dict)

    #arch_elements = _neurons + _routers
    arch_elements = _neurons + _routers + _memories + _synapses + \
                    _axon_inputs + _axon_outputs + _dendrites + \
                    _timers
    #print(arch_elements)

    with open("out", "w") as list_file:
        for line in arch_elements:
            print(line) 
            list_file.write(line + '\n')
