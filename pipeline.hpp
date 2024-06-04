// Copyright (c) 2024 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
//  pipeline.hpp
namespace sanafe
{
struct Architecture;
struct Timestep;

void pipeline_process_neurons(Timestep &ts, Architecture &arch);
void pipeline_receive_messages(Timestep &ts, Architecture &arch);

void pipeline_process_neuron(Timestep &ts, Architecture &arch, Neuron &n);
double pipeline_process_message(Timestep &ts, Architecture &arch, Core &c, Message &m);

void pipeline_neuron_send_spike_messages(Timestep &ts, Architecture &arch, Neuron &n);
double pipeline_update_synapse(Timestep &ts, Architecture &arch, Core &c, const int synapse_address);
double pipeline_update_dendrite(Timestep &ts, Architecture &arch, Neuron &n);
double pipeline_update_soma(Timestep &ts, Architecture &arch, Neuron &n);
}