// Copyright (c) 2024 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
// network.h - (spiking) neural network functionality. Spiking neural
//  networks are represented as groups of neurons. A neuron group might have a
//  bunch of neurons all with the same properties (and common hardware).
//  Each neuron has its own state and a set of connections to other neurons.
//  These structures have links to hardware for performance simulation.
//  Here we include different neuron, synapse and dendrite models.
#ifndef NETWORK_HEADER_INCLUDED_
#define NETWORK_HEADER_INCLUDED_

#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <functional> // For std::reference_wrapper
#include "plugins.hpp"
#include "models.hpp"

namespace sanafe
{
enum ConnectionConfigFormat
{
	// This is the structure of the CSV format for specifying synaptic
	//  connections in a network for this simulator.  Each row represents
	//  a unique connection.
	CONNECTION_DEST_GID = 0,
	CONNECTION_DEST_NID,
	CONNECTION_WEIGHT,
	CONNECTION_FIELDS,
};

// Forward declarations
class Core;
struct Neuron;
struct NeuronGroup;
class Architecture;
struct SomaUnit;
struct SomaModel;
struct SynapseUnit;
struct AxonOutUnit;

struct Connection
{
	Neuron *post_neuron, *pre_neuron;
	SynapseUnit *synapse_hw;
	std::string synapse_hw_name;
	double weight, current, synaptic_current_decay;
	int id, delay, last_updated;

	Connection(const int connection_id);
};

struct Neuron
{
	std::vector<Connection> connections_out;
	std::vector<int> axon_out_addresses;
	std::unordered_map<std::string, std::string> attributes;

	// Mapped hardware
	Network *parent_net;
	Core *core, *post_synaptic_cores;
	SomaUnit *soma_hw;
	AxonOutUnit *axon_out_hw;
	std::string soma_hw_name;

	std::shared_ptr<SomaModel> model;

	// Track the timestep each hardware unit was last updated
	bool fired, force_update, log_spikes, log_potential;
	bool update_needed;
	int id, parent_group_id;
	int spike_count;
	int soma_last_updated, dendrite_last_updated;
	int max_connections_out, maps_in_count, maps_out_count;

	double dendritic_current_decay, processing_latency;
	double current, charge;
	NeuronStatus neuron_status;
	int forced_spikes;

	Neuron(const size_t neuron_id);
	int get_id() { return id; }
	void set_attributes(const std::unordered_map<std::string, std::string> &attr);
	void connect_to_neuron(Neuron &dest, const std::unordered_map<std::string, std::string> &attr);
};

class NeuronGroup
{
public:
	// A neuron group is a collection of neurons that share common
	//  parameters. All neurons must be based on the same neuron model.
	std::vector<Neuron> neurons;
	std::string default_soma_hw_name;
	std::string default_synapse_hw_name;
	std::unordered_map<std::string, std::string> default_attributes;

	int id;
	int default_max_connections_out;
	bool default_log_potential, default_log_spikes, default_force_update;

	int get_id() { return id; }
	NeuronGroup(const size_t group_id, const int neuron_count);
	void set_attribute_multiple(const std::string &attr, const std::vector<std::string> &values);
	void connect_neurons(NeuronGroup &dest_group, const std::vector<std::pair<int, int> > &src_dest_id_pairs, const std::unordered_map<std::string, std::vector<std::string> > &attr_lists);
};

class Network
{
public:
	std::list<NeuronGroup> groups;
	std::vector<std::reference_wrapper<NeuronGroup> > groups_vec;
	Network() {};
	NeuronGroup &create_neuron_group(const int neuron_count, const std::unordered_map<std::string, std::string> &attr);
	void load_net_file(const std::string &filename, Architecture &arch);
private:
	// Do *NOT* allow Network objects to be copied
	//  This is because Neuron objects link back to their parent Network
	//  (and need to be aware of the parent Group). Linking to parent
	//  objects allows us to efficiently store Attributes for neurons i.e.,
	//  by avoiding duplication of shared attributes.
	//  If the Network was moved or copied, all parent links would be
	//  invalidated.
	Network(const Network &copy);
};

void network_check_mapped(Network &net);
}

#endif
