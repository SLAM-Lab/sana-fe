// Copyright (c) 2023 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
//  sim.c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "print.h"
#include "sim.h"
#include "network.h"
#include "arch.h"

struct timestep sim_timestep(struct simulation *const sim,
	struct network *const net, struct architecture *const arch)
{
	struct timestep ts;

	// Start the next time-step
	sim_init_timestep(&ts);
	sim_reset_measurements(net, arch);

	ts.timestep = sim->timesteps;
	sim_send_messages(sim, &ts, net, arch);
	sim_input_spikes(net);
	sim_receive_messages(&ts, arch);
	sim_schedule_messages(sim, &ts, arch);

	// Performance statistics for this time step
	ts.sim_time = sim_calculate_time(arch);
	ts.energy = sim_calculate_energy(arch);
	for (int i = 0; i < arch->tile_count; i++)
	{
		struct tile *t = &(arch->tiles[i]);
		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);
			ts.spike_count += c->synapse.spikes_processed;
			ts.total_neurons_fired += c->soma.neurons_fired;

		}
		ts.messages_sent += t->messages_received;
	}
	if (sim->perf_fp)
	{
		sim_perf_log_timestep(&ts, sim->perf_fp);
	}

	sim->timesteps++;
	sim->total_energy += ts.energy;
	sim->total_sim_time += ts.sim_time;
	sim->total_spikes += ts.spike_count;
	sim->total_neurons_fired += ts.total_neurons_fired;
	sim->total_messages_sent += ts.messages_sent;

	TRACE1("Spikes sent: %ld\n", sim->total_spikes);
	return ts;
}

void sim_init_sim(struct simulation *sim)
{
	sim->total_energy = 0.0;   // Joules
	sim->total_sim_time = 0.0; // Seconds
	sim->wall_time = 0.0;	   // Seconds
	sim->timesteps = 0;
	sim->total_spikes = 0;
	sim->total_messages_sent = 0;
	sim->total_neurons_fired = 0;

	// All logging disabled by default
	sim->log_perf = 0;
	sim->log_potential = 0;
	sim->log_spikes = 0;
	sim->log_messages = 0;

	sim->potential_trace_fp = NULL;
	sim->spike_trace_fp = NULL;
	sim->perf_fp = NULL;
	sim->message_trace_fp = NULL;
	sim->stats_fp = NULL;
}

void sim_init_timestep(struct timestep *const ts)
{
	ts->timestep = -1;
	ts->spike_count = 0;
	ts->messages_sent = 0;
	ts->total_neurons_fired = 0;
	ts->spikes = 0;
	ts->total_hops = 0;
	ts->energy = 0.0;
	ts->sim_time = 0.0;
}

void sim_send_messages(struct simulation *const sim, struct timestep *const ts,
	struct network *net, struct architecture *arch)
{
#pragma omp parallel for
	for (int i = 0; i < arch->tile_count; i++)
	{
		struct tile *t = &(arch->tiles[i]);

		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);
			for (int k = 0; k < c->neuron_count; k++)
			{
				struct neuron *n = c->neurons[k];
				assert(n != NULL);
				sim_process_neuron(ts, n);
			}
		}
	}
	if (sim->log_spikes)
	{
		sim_trace_record_spikes(sim, net);
	}
	if (sim->log_potential)
	{
		sim_trace_record_potentials(sim, net);
	}
}

void sim_receive_messages(struct timestep *const ts, struct architecture *arch)
{
#pragma omp parallel for
	for (int i = 0; i < arch->tile_count; i++)
	{
		struct tile *t = &(arch->tiles[i]);

		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);
			for (int k = 0; k < c->axon_in.map_count; k++)
			{
				struct connection_map *axon =
					&(c->axon_in.map[k]);
				if (axon->spikes_received > 0)
				{
					struct neuron *pre_neuron =
						axon->pre_neuron;
					assert(pre_neuron != NULL);
					struct core *pre_core =
						pre_neuron->core;
					assert(pre_core != NULL);
					struct tile *pre_tile = pre_core->t;
					assert(pre_tile != NULL);
					axon->network_latency =
						sim_estimate_network_costs(
							pre_tile, t);
					axon->receive_latency =
						sim_pipeline_receive(
							ts, c, axon);
				}
			}
		}
	}
}

double sim_estimate_network_costs(struct tile *const src,
	struct tile *const dest)
{
	double network_latency;
	long int x_hops, y_hops;

	assert(src != NULL);
	assert(dest != NULL);

	dest->energy += src->energy_spike_within_tile;
	network_latency = src->time_spike_within_tile;

	// Calculate the energy and time for sending spike packets
	x_hops = abs(src->x - dest->x);
	y_hops = abs(src->y - dest->y);
	// E-W hops
	dest->energy += (double) x_hops * src->energy_east_west_hop;
	network_latency += (double) x_hops * src->time_east_west_hop;
	// N-S hops
	dest->energy += (double) y_hops * src->energy_north_south_hop;
	network_latency += (double) y_hops * src->time_north_south_hop;

	dest->hops += (x_hops + y_hops);
	dest->messages_received++;
	TRACE1("xhops:%ld yhops%ld total hops:%ld latency:%e\n", x_hops, y_hops,
		t->hops, network_latency);
	return network_latency;
}

void sim_init_message(struct message *const m)
{
	m->src_neuron = NULL;
	m->src_tile = NULL;
	m->src_core = NULL;
	m->dest_tile = NULL;
	m->dest_core = NULL;
	m->generation_latency = 0.0;
	m->network_latency = 0.0;
	m->receive_latency = 0.0;
}

int sim_schedule_messages(const struct simulation *const sim,
	struct timestep *const ts, struct architecture *const arch)
{
	struct core *top_priority;
	const int core_count = ARCH_MAX_CORES * ARCH_MAX_TILES;
	int cores_left;

	top_priority = sim_init_timing_priority(arch);

	// Setup timing counters
	TRACE1("Scheduling global order of messages.\n");
	cores_left = core_count;
	for (int i = 0; i < arch->tile_count; i++)
	{
		struct tile *t = &(arch->tiles[i]);
		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);
			c->neurons_left = c->neuron_count;
			c->curr_neuron = 0;
			c->messages_left = 0;
			c->curr_axon = 0;
		}
	}

	while (cores_left && (top_priority != NULL))
	{
		// Get the core with the earliest simulation time
		struct message m;
		struct core *c = top_priority;

		// If all messages were processed, get the next spiking neuron
		//  and see how many spike messages to send
		sim_init_message(&m);
		if (c->messages_left <= 0)
		{
			// Get the next spiking neuron that sends one or more
			//  spike messages
			TRACE2("Getting next firing neuron for cid:%d\n",
				c->id);
			while (c->neurons_left > 0)
			{
				int nid = c->curr_neuron;
				assert(nid < c->neuron_count);
				m.src_neuron = c->neurons[nid];
				assert(m.src_neuron != NULL);

				c->time += m.src_neuron->processing_latency;
				m.generation_latency +=
					m.src_neuron->processing_latency;
				if (!(m.src_neuron->is_init) ||
					!(m.src_neuron->fired))
				{
					c->curr_neuron++;
					c->neurons_left--;
					c->curr_axon = 0;
				}
				else if (m.src_neuron->fired)
				{
					TRACE2("(cid:%d.%d) nid:%d.%d fired, "
					       "time:%e\n",
						c->t->id, c->id,
						m.src_neuron->group->id,
						m.src_neuron->id, c->time);
					ts->total_neurons_fired++;
					c->messages_left =
						m.src_neuron->maps_out_count;
					c->curr_axon = 0;

					break;
				}
			}
		}
		else
		{
			// Schedule this neuron's messages
			struct neuron *n;
			int nid = c->curr_neuron;

			m.src_neuron = c->neurons[nid];
			assert(m.src_neuron->maps_out != NULL);
			assert(m.src_neuron->maps_out[0] != NULL);

			m.dest_axon = m.src_neuron->maps_out[c->curr_axon];
			assert(m.dest_axon != NULL);
			assert(m.dest_axon->connections != NULL);
			assert(m.dest_axon->spikes_received > 0);
			m.spikes = m.dest_axon->connection_count;
			m.src_core = c;
			m.src_tile = c->t;
			assert(m.src_tile != NULL);

			n = m.dest_axon->connections[0]->post_neuron;
			m.receive_latency = m.dest_axon->receive_latency;
			m.network_latency = m.dest_axon->network_latency;
			m.dest_core = n->core;
			m.dest_tile = m.dest_core->t;
			assert(m.dest_tile != NULL);
			m.hops = abs(m.src_tile->x - m.dest_tile->x);
			m.hops += abs(m.src_tile->y - m.dest_tile->y);

			TRACE1("\t(cid:%d.%d) Sending %d spikes to %d.%d\n",
				c->t->id, c->id, m.dest_axon->connection_count,
				m.dest_core->t->id, m.dest_core->id);
			TRACE1("cid:%d: %d messages left\n", c->id,
				c->messages_left);

			// Add axon access cost to message latency and energy
			c->time += c->axon_out.time_access;
			m.generation_latency += c->axon_out.time_access;
			c->axon_out.energy += c->axon_out.energy_access;

			// Account for core hardware interactions, blocking the
			//  message from being scheduled if the receiving h/w
			//  is busy
			m.blocked_latency = 0.0;
			if (m.dest_tile->is_blocking)
			{
				m.blocked_latency = fmax(m.blocked_latency,
					m.dest_tile->blocked_until - c->time);
				// Update the core global time, blocking until
				//  the receiving tile is free
				c->time = fmax(
					c->time, m.dest_tile->blocked_until);
			}
			if (m.dest_core->is_blocking)
			{
				// Track how long the message is blocked for
				m.blocked_latency = fmax(m.blocked_latency,
					m.dest_core->blocked_until - c->time);
				// Calculate when the receiving h/w will be busy
				//  until
				if (c->time < m.dest_core->blocked_until)
				{
					// If we were trying to send a spike to
					//  a blocked core, also block the tile
					//  for this duration as well
					m.dest_tile->blocked_until =
						m.dest_core->blocked_until;
					// Update the core global time, blocking
					//  until the receiving core is free
					c->time = m.dest_core->blocked_until;
				}
			}

			c->time += m.dest_axon->network_latency;
			if (m.dest_core->is_blocking)
			{
				m.dest_core->blocked_until =
					c->time + m.receive_latency;
			}
			else
			{
				m.dest_core->blocked_until += m.receive_latency;
			}
			TRACE2("\t(cid:%d.%d) synapse at %d.%d busy until %e\n",
				c->t->id, c->id, m.dest_core->t->id,
				m.dest_core->id, m.dest_core->blocked_until);

			// Get the next message
			c->curr_axon++;
			c->messages_left--;
			TRACE2("cid:%d axon:%d (%d messages left)\n", c->id,
				c->curr_axon, c->messages_left);
			if (sim->log_messages)
			{
				sim_trace_record_message(sim, &m);
			}
			sim_init_message(&m);
		}

		// Get the next message, neuron or core
		if (c->neurons_left && c->messages_left <= 0)
		{
			// We just scheduled the last message, get the next
			//  neuron to process
			c->curr_neuron++;
			c->neurons_left--;
		}
		if (c->neurons_left <= 0)
		{
			TRACE1("\t(cid:%d.%d) finished simulating\n", c->t->id,
				c->id);
			// The last neuron's messages are all scheduled, so
			//  remove the core from the priority queue
			cores_left--;
			assert(cores_left >= 0);
			top_priority = top_priority->next_timing;
		}

		top_priority = sim_update_timing_queue(top_priority);
		if (top_priority != NULL)
		{
			TRACE2("\t(cid:%d.%d) time:%e\n", top_priority->t->id,
				top_priority->id, top_priority->time);
		}
	}
	TRACE1("Neurons fired: %ld\n", ts->total_neurons_fired);

	return 0;
}

void sim_process_neuron(struct timestep *const ts, struct neuron *n)
{
	if (!n->is_init)
	{
		return;
	}

	struct core *c = n->core;
	n->processing_latency = 0.0;

	// TODO: I think there's a flaw here. We update synapses for the core,
	//  not for the neuron. Then each neuron does have a single dendritic
	//  tree. So it's just this, maybe we make it "process neurons"
	if (c->buffer_pos == BUFFER_SYNAPSE)
	{
		for (int i = 0; i < n->maps_in_count; i++)
		{
			struct connection_map *axon = &(n->maps_in[i]);
			n->processing_latency +=
				sim_update_synapse(ts, axon, 1);
		}
	}
	else if (c->buffer_pos == BUFFER_DENDRITE)
	{
		// Go through all synapses connected to this neuron and update
		//  all synaptic currents into the dendrite
		for (int i = 0; i < n->maps_in_count; i++)
		{
			struct connection_map *a = &(n->maps_in[i]);
			for (int j = 0; j < a->connection_count; j++)
			{
				struct connection *con = a->connections[j];
				n->processing_latency += sim_update_dendrite(
					ts, n, con->current);
			}
		}
	}
	else if (c->buffer_pos == BUFFER_SOMA)
	{
		n->processing_latency = sim_update_soma(ts, n, n->charge);
	}
	else if (c->buffer_pos == BUFFER_AXON_OUT)
	{
		if (n->fired)
		{
			n->processing_latency = sim_neuron_send_spike(n);
		}
	}
	TRACE1("Updating neuron %d.%d.\n", n->group->id, n->id);

	n->update_needed = 0;
	n->spike_count = 0;
}

double sim_pipeline_receive(
	struct timestep *const ts, struct core *c, struct connection_map *axon)
{
	// We receive a spike and process up to the time-step buffer
	double message_processing_latency = 0.0;

	TRACE1("Receiving messages for cid:%d\n", c->id);
	if (c->buffer_pos >= BUFFER_SYNAPSE)
	{
		int synaptic_lookup = 1;
		message_processing_latency =
			sim_update_synapse(ts, axon, synaptic_lookup);
	}

	return message_processing_latency;
}

struct core *sim_init_timing_priority(struct architecture *arch)
{
	struct core *next = NULL;

	// Initialize in reverse order, but assuming all cores start time
	//  synchronized (time ==), this is arbitrary
	for (int i = (arch->tile_count - 1); i >= 0; i--)
	{
		struct tile *t = &(arch->tiles[i]);
		for (int j = (t->core_count - 1); j >= 0; j--)
		{
			struct core *c = &(t->cores[j]);
			assert(c->time == 0.0);
			if (c->neuron_count > 0)
			{
				c->next_timing = next;
				next = c;
			}
		}
	}

	for (struct core *curr = next; curr != NULL; curr = curr->next_timing)
	{
		TRACE2("curr:%d.%d\n", curr->t->id, curr->id);
	}

	return next;
}

struct core *sim_update_timing_queue(struct core *top_priority)
{
	struct core *next, *tmp;
	int list_depth = 0;

	if ((top_priority == NULL) || (top_priority->next_timing == NULL) ||
		(top_priority->time <= top_priority->next_timing->time))
	{
		// Do nothing
		return top_priority;
	}
	else
	{
		struct core *curr = top_priority;
		next = top_priority->next_timing;

		tmp = top_priority;
		top_priority = next;
		// reinsert core into the correct place in the priority list
		while (next != NULL)
		{
			if (tmp->time < next->time)
			{
				break;
			}
			curr = next;
			next = curr->next_timing;
			list_depth++;
		}
		curr->next_timing = tmp;
		tmp->next_timing = next;
	}

	for (struct core *tmp = top_priority; tmp != NULL;
		tmp = tmp->next_timing)
	{
		TRACE3("tmp->time:%e (id:%d)\n", tmp->time, tmp->id);
		assert((tmp->next_timing == NULL) ||
			tmp->time <= tmp->next_timing->time);
	}
	TRACE2("List depth = %d\n", list_depth);

	return top_priority;
}

int sim_input_spikes(struct network *net)
{
	int input_spike_count = 0;

	// Seed all externally input spikes in the network for this timestep
	for (int i = 0; i < net->external_input_count; i++)
	{
		struct input *in = &(net->external_inputs[i]);

		if (in == NULL)
		{
			break;
		}

		if (in->type == INPUT_EVENT)
		{
			in->send_spike = (in->spike_val > 0.0);
		}
		else if (in->type == INPUT_POISSON)
		{
			in->send_spike = sim_poisson_input(in->rate);
		}
		else // INPUT_RATE)
		{
			in->send_spike =
				sim_rate_input(in->rate, &(in->spike_val));
		}

		if (!in->send_spike)
		{
			TRACE3("Not sending spike\n");
			continue;
		}
		for (int j = 0; j < in->post_connection_count; j++)
		{
			// Send a spike to all neurons connected to this input
			//  Normally, we would have a number of input dimensions
			//  for a given network
			struct neuron *post_neuron;
			struct connection *connection_ptr;

			connection_ptr = &(in->connections[j]);
			assert(connection_ptr);

			post_neuron = connection_ptr->post_neuron;
			TRACE3("nid:%d Energy before: %lf\n", post_neuron->id,
				post_neuron->current);
			if (post_neuron->core->buffer_pos == BUFFER_SOMA)
			{
				post_neuron->charge += connection_ptr->weight;
			}
			else
			{
				post_neuron->current += connection_ptr->weight;
			}
			TRACE3("nid:%d Energy after: %lf\n", post_neuron->id,
				post_neuron->current);

			post_neuron->synapse_hw->energy +=
				post_neuron->synapse_hw->energy_spike_op;
			post_neuron->synapse_hw->time +=
				post_neuron->synapse_hw->time_spike_op;

			post_neuron->update_needed = 1;
			post_neuron->spike_count++;
			input_spike_count++;
		}
		TRACE1("Sent spikes to %d connections\n",
			in->post_connection_count);

		// If inputting sets of events, then reset the spike after
		//  it's processed i.e. inputs are one-shot. For poisson and
		//  rate inputs, their values stay unchanged until the user
		//  sets them again (e.g. until the next input is presented)
		if (in->type == INPUT_EVENT)
		{
			in->spike_val = 0.0;
		}
	}

	return input_spike_count;
}

double sim_update_synapse(struct timestep *const ts,
	struct connection_map *axon, const int synaptic_lookup)
{
	// Update all synapses to different neurons in one core. If a synaptic
	//  lookup, read and accumulate the synaptic weights. Otherwise, just
	//  update filtered current and any other connection properties
	struct core *post_core;
	double latency, min_synaptic_resolution;
	int spike_ops, memory_accesses;

	latency = 0.0;
	post_core = axon->connections[0]->post_neuron->core;
	spike_ops = axon->connection_count;

	TRACE1("Updating synapses for (cid:%d)\n", axon->pre_neuron->id);
	while (axon->last_updated <= ts->timestep)
	{
		TRACE1("Updating synaptic current (last updated:%ld, ts:%ld)\n",
			axon->last_updated, ts->timestep);
		if (axon->active_synapses > 0)
		{
			for (int i = 0; i < axon->connection_count; i++)
			{
				struct neuron *post_neuron;
				struct connection *con = axon->connections[i];

				post_neuron = con->post_neuron;
				con->current *= con->synaptic_current_decay;

				// "Turn off" synapses that have basically no
				//  synaptic current left to decay (based on
				//  the weight resolution)
				min_synaptic_resolution =
					1.0 / post_core->synapse.weight_bits;
				if (fabs(con->current) <
					min_synaptic_resolution)
				{
					con->current = 0.0;
					axon->active_synapses--;
				}

				TRACE2("(nid:%d->nid:%d) con->current:%lf\n",
					con->pre_neuron->id,
					con->post_neuron->id, con->current);
				if (post_core->buffer_pos != BUFFER_DENDRITE)
				{
					latency += sim_update_dendrite(
						ts, post_neuron, con->current);
				}
			}
		}
		axon->last_updated++;
	}

	if (synaptic_lookup)
	{
		axon->active_synapses = axon->connection_count;
		// Simple memory access model where we consider the number of
		//  words accessed and the cost per access
		memory_accesses =
			(spike_ops +
				(post_core->synapse.weights_per_word - 1)) /
			(post_core->synapse.weights_per_word);
		TRACE2("weights_per_word:%d memory_acceses:%d\n",
			post_core->synapse.weights_per_word, memory_accesses);
		post_core->synapse.energy +=
			memory_accesses *
			post_core->synapse.energy_memory_access;
		latency +=
			memory_accesses * post_core->synapse.time_memory_access;

		post_core->synapse.energy +=
			spike_ops * post_core->synapse.energy_spike_op;
		// TODO: generalize, number of parallel processing units
		if (spike_ops <= 4)
		{
			latency += 12.0e-9;
		}

		latency += floor(((double) spike_ops + 2.0) / 3.0) *
				(post_core->synapse.time_spike_op * 3.0);

		post_core->synapse.spikes_processed += spike_ops;
		post_core->synapse.memory_reads += memory_accesses;

		for (int i = 0; i < axon->connection_count; i++)
		{
			struct neuron *post_neuron;
			struct connection *con = axon->connections[i];

			con->current += con->weight;
			post_neuron = con->post_neuron;
			post_neuron->update_needed = 1;
			post_neuron->spike_count++;
			TRACE2("Sending spike to nid:%d, current:%lf\n",
				post_neuron->id, con->current);

			if (post_core->buffer_pos != BUFFER_DENDRITE)
			{
				latency += sim_update_dendrite(
					ts, post_neuron, con->current);
			}
		}
	}

	return latency;
}

double sim_update_dendrite(
	struct timestep *const ts, struct neuron *n, const double charge)
{
	// TODO: Support dendritic operations, combining the current in
	//  different neurons in some way, and writing the result to an output
	double dendritic_current, latency;
	latency = 0.0;

	dendritic_current = 0.0;
	while (n->dendrite_last_updated <= ts->timestep)
	{
		TRACE3("Updating dendritic current (last_updated:%d, ts:%ld)\n",
			n->dendrite_last_updated, ts->timestep);
		n->charge *= n->dendritic_current_decay;
		n->dendrite_last_updated++;
		dendritic_current = n->charge;
		TRACE2("nid:%d charge:%lf\n", n->id, n->charge);
	}

	// Update dendritic tap currents
	// TODO: implement multi-tap models
	TRACE2("Charge:%lf\n", charge);
	dendritic_current += charge;
	n->charge += charge;

	// Finally, send dendritic current to the soma
	TRACE2("nid:%d updating dendrite, charge:%lf\n", n->id, n->charge);
	if (n->core->buffer_pos != BUFFER_SOMA)
	{
		latency += sim_update_soma(ts, n, dendritic_current);
	}

	return latency;
}

double sim_update_soma(
	struct timestep *const ts, struct neuron *n, const double current_in)
{
	double latency = 0.0;
	struct soma_processor *soma = n->soma_hw;

	if (!n->update_needed)
	{
		// Inactive neuron update cost
		TRACE2("nid:%d not updated.\n", n->id);
		latency += n->soma_hw->time_inactive_neuron_update;
		n->soma_hw->energy += n->soma_hw->energy_inactive_neuron_update;
		return latency;
	}

	TRACE1("nid:%d updating, current_in:%lf\n", n->id, current_in);
	if (soma->model == NEURON_LIF)
	{
		latency += sim_update_soma_lif(ts, n, current_in);
	}
	else if (soma->model == NEURON_TRUENORTH)
	{
		latency += sim_update_soma_truenorth(ts, n, current_in);
	}
	else
	{
		INFO("Neuron model not recognised: %d", soma->model);
		assert(0);
	}

	return latency;
}

double sim_generate_noise(struct core *c)
{
	int noise_val = 0;
	int ret;

	if (c->noise_type == NOISE_FILE_STREAM)
	{
		// With a noise stream, we have a file containing a series of
		//  random values. This is useful if we want to exactly
		//  replicate h/w without knowing how the stream is generated.
		//  We can record the random sequence and replicate it here
		char noise_str[MAX_NOISE_FILE_ENTRY];
		// If we get to the end of the stream, by default reset it.
		//  However, it is unlikely the stream will be correct at this
		//  point
		if (feof(c->noise_stream))
		{
			INFO("Warning: At the end of the noise stream. "
			     "Random values are unlikely to be correct.\n");
			fseek(c->noise_stream, 0, SEEK_SET);
		}
		fgets(noise_str, MAX_NOISE_FILE_ENTRY, c->noise_stream);
		ret = sscanf(noise_str, "%d", &noise_val);
		TRACE2("noise val:%d\n", noise_val);
		if (ret < 1)
		{
			INFO("Error: invalid noise stream entry.\n");
		}
	}

	// Get the number of noise bits required TODO: generalize
	int sign_bit = noise_val & 0x100;
	noise_val &= 0x7f; // TODO: hack, fixed for 8 bits
	if (sign_bit)
	{
		// Sign extend
		noise_val |= ~(0x7f);
	}

	return (double) noise_val;
}

double sim_update_soma_lif(
	struct timestep *const ts, struct neuron *n, const double current_in)

{
	struct soma_processor *soma = n->soma_hw;
	double random_potential, latency = 0.0;

	// Calculate the change in potential since the last update e.g.
	//  integate inputs and apply any potential leak
	TRACE1("Updating potential, before:%f\n", n->potential);

	if (soma->model == NEURON_LIF)
	{
		while (n->soma_last_updated <= ts->timestep)
		{
			n->potential *= n->leak_decay;
			n->soma_last_updated++;
		}
	}

	// Add randomized noise to potential if enabled
	if (n->core->noise_type == NOISE_FILE_STREAM)
	{
		random_potential = sim_generate_noise(n->core);
		n->potential += random_potential;
	}

	// Add the synaptic / dendrite current to the potential
	//printf("n->bias:%lf n->potential before:%lf current_in:%lf\n", n->bias, n->potential, current_in);
	n->potential += current_in + n->bias;
	n->charge = 0.0;
	//printf("n->bias:%lf n->potential after:%lf\n", n->bias, n->potential);

	TRACE1("Updating potential, after:%f\n", n->potential);

	// Check against threshold potential (for spiking)
	if (((n->bias != 0.0) && (n->potential > n->threshold)) ||
		((n->bias == 0.0) && (n->potential >= n->threshold)))
	{
		if (n->group->reset_mode == NEURON_RESET_HARD)
		{
			n->potential = n->reset;
		}
		else if (n->group->reset_mode == NEURON_RESET_SOFT)
		{
			n->potential -= n->threshold;
		}
		latency += sim_neuron_send_spike(n);
		soma->neurons_fired++;
	}

	// Check against reverse threshold
	if (n->potential < n->reverse_threshold)
	{
		if (n->group->reverse_reset_mode == NEURON_RESET_SOFT)
		{
			n->potential -= n->reverse_threshold;
		}
		else if (n->group->reverse_reset_mode == NEURON_RESET_HARD)
		{
			n->potential = n->reverse_reset;
		}
		else if (n->group->reverse_reset_mode == NEURON_RESET_SATURATE)
		{
			n->potential = n->reverse_threshold;
		}
	}

	// Update soma, if there are any received spikes, there is a non-zero
	//  bias or we force the neuron to update every time-step
	if (n->spike_count || (fabs(n->bias) > 0.0) || n->force_update)
	{
		latency += n->soma_hw->time_active_neuron_update;
		soma->updates++;
		n->soma_hw->energy += n->soma_hw->energy_active_neuron_update;
	}

	return latency;
}

double sim_update_soma_truenorth(
	struct timestep *const ts, struct neuron *n, const double current_in)
{
	struct soma_processor *soma = n->soma_hw;
	double v, latency = 0.0;
	int randomize_threshold;

	// Apply leak
	while (n->soma_last_updated <= ts->timestep)
	{
		// Linear leak
		if (soma->leak_towards_zero)
		{
			// TODO: what happens if we're above zero but by less
			//  than the leak amount (for convergent), will we
			//  oscillate between the two? Does it matter
			if (n->potential > 0.0)
			{
				n->potential -= n->leak_bias;
			}
			else if (n->potential < 0.0)
			{
				n->potential += n->leak_bias;
			}
			// else equals zero, so no leak is applied
		}
		else
		{
			n->potential += n->leak_decay;
		}
		n->soma_last_updated++;
	}

	// Add the synaptic currents, processed by the dendrite
	n->potential += current_in + n->bias;
	n->current = 0.0;
	n->charge = 0.0;

	// Apply thresholding and reset
	v = n->potential;
	randomize_threshold = (n->random_range_mask != 0);
	if (randomize_threshold)
	{
		unsigned int r = rand() & n->random_range_mask;
		v += (double) r;
	}

	TRACE2("v:%lf +vth:%lf mode:%d -vth:%lf mode:%d\n", v, n->threshold,
		n->group->reset_mode, n->reverse_threshold,
		n->group->reverse_reset_mode);
	if (v >= n->threshold)
	{
		int reset_mode = n->group->reset_mode;
		if (reset_mode == NEURON_RESET_HARD)
		{
			n->potential = n->reset;
		}
		else if (reset_mode == NEURON_RESET_SOFT)
		{
			n->potential -= n->threshold;
		}
		else if (reset_mode == NEURON_RESET_SATURATE)
		{
			n->potential = n->threshold;
		}
		latency += sim_neuron_send_spike(n);
	}
	else if (v <= n->reverse_threshold)
	{
		int reset_mode = n->group->reverse_reset_mode;
		if (reset_mode == NEURON_RESET_HARD)
		{
			n->potential = n->reverse_reset;
		}
		else if (reset_mode == NEURON_RESET_SOFT)
		{
			n->potential += n->reverse_threshold;
		}
		else if (reset_mode == NEURON_RESET_SATURATE)
		{
			n->potential = n->reverse_threshold;
		}
		// No spike is generated
	}
	TRACE2("potential:%lf threshold %lf\n", n->potential, n->threshold);

	return latency;
}

double sim_neuron_send_spike(struct neuron *n)
{
	struct soma_processor *soma = n->soma_hw;
	double latency = 0.0;

	latency += soma->time_spiking;
	soma->energy += soma->energy_spiking;
	if (n->core->buffer_pos != BUFFER_AXON_OUT)
	{
		TRACE1("nid:%d fired.\n", n->id);
		n->fired = 1;
		// Record a spike message at all the connected cores (axons)
		for (int k = 0; k < n->maps_out_count; k++)
		{
			n->maps_out[k]->spikes_received++;
		}
	}

	return latency;
}

double sim_calculate_time(const struct architecture *const arch)
{
	// Returns the simulation time of the current timestep.
	//  This is calculated by finding the simulation time of each core,
	//  and simply taking the maximum of this.
	double max_time = 0.0;

	for (int i = 0; i < arch->tile_count; i++)
	{
		const struct tile *t = &(arch->tiles[i]);
		double max_core_time = 0.0;
		for (int j = 0; j < t->core_count; j++)
		{
			const struct core *c = &(t->cores[j]);
			double this_core_time = 0.0;
			this_core_time = fmax(c->time, c->blocked_until);
			max_core_time = fmax(max_core_time, this_core_time);
		}

		max_time = fmax(max_time, max_core_time);
	}

	// Add the mesh-wide barrier sync time (assuming worst case of 32 tiles)
	max_time += arch->time_barrier;
	//INFO("Simulated time for step is:%es\n", max_time);

	return max_time;
}

double sim_calculate_energy(const struct architecture *const arch)
{
	// Returns the total energy across the design, for this timestep
	double total_energy, dendrite_energy, soma_energy, synapse_energy;
	double axon_energy, network_energy;

	synapse_energy = 0.0;
	dendrite_energy = 0.0;
	soma_energy = 0.0;
	axon_energy = 0.0;
	network_energy = 0.0;

	for (int i = 0; i < arch->tile_count; i++)
	{
		const struct tile *t = &(arch->tiles[i]);
		network_energy += t->energy;

		for (int j = 0; j < t->core_count; j++)
		{
			const struct core *c = &(t->cores[j]);
			synapse_energy += c->synapse.energy;
			soma_energy += c->soma.energy;
			axon_energy += c->axon_out.energy;
		}
	}

	total_energy = synapse_energy + dendrite_energy + soma_energy +
		       axon_energy + network_energy;
	return total_energy;
}

void sim_reset_measurements(struct network *net, struct architecture *arch)
{
	for (int i = 0; i < net->neuron_group_count; i++)
	{
		struct neuron_group *group = &(net->groups[i]);

		for (int j = 0; j < group->neuron_count; j++)
		{
			struct neuron *n = &(group->neurons[j]);
			// Neurons can be manually forced to update, for example
			//  if they have a constant input bias
			n->update_needed |=
				(n->force_update || (n->bias != 0.0));
			n->processing_latency = 0.0;
			n->fired = 0;

			for (int k = 0; k < n->maps_out_count; k++)
			{
				struct connection_map *a = n->maps_out[k];
				a->network_latency = 0.0;
				a->receive_latency = 0.0;
				a->spikes_received = 0;
			}
		}
	}

	// Reset any energy, time latency or other measurements of network
	//  hardware
	for (int i = 0; i < arch->tile_count; i++)
	{
		struct tile *t = &(arch->tiles[i]);
		// Reset tile
		t->energy = 0.0;
		t->time = 0.0;
		t->blocked_until = 0.0;

		t->hops = 0;
		t->messages_received = 0;
		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);
			// Reset core
			c->time = 0.0;
			c->energy = 0.0;
			c->blocked_until = 0.0;
			c->neurons_left = 0;
			c->messages_left = 0;
			c->curr_axon = 0;
			c->curr_neuron = 0;

			c->axon_in.energy = 0.0;
			c->axon_in.time = 0;

			c->synapse.energy = 0.0;
			c->synapse.time = 0.0;
			c->synapse.spikes_processed = 0;

			c->dendrite.energy = 0.0;
			c->dendrite.time = 0.0;

			c->soma.energy = 0.0;
			c->soma.time = 0.0;
			c->soma.neurons_fired = 0;

			c->axon_out.energy = 0.0;
			c->axon_out.time = 0.0;
			c->axon_out.packets_out = 0;
		}
	}
}

void sim_perf_write_header(FILE *fp)
{
	fprintf(fp, "time,");
	fprintf(fp, "fired,");
	fprintf(fp, "packets,");
	fprintf(fp, "hops,");
	fprintf(fp, "total_energy,");
	fprintf(fp, "\n");
}

void sim_perf_log_timestep(const struct timestep *const ts, FILE *fp)
{
	fprintf(fp, "%le,", ts->sim_time);
	fprintf(fp, "%ld,", ts->total_neurons_fired);
	fprintf(fp, "%ld,", ts->messages_sent);
	fprintf(fp, "%ld,", ts->total_hops);
	fprintf(fp, "%le,", ts->energy);
	fprintf(fp, "\n");
}

void sim_write_summary(FILE *fp, const struct simulation *sim)
{
	// Write the simulation summary to file
	fprintf(fp, "git_version: %s\n", GIT_COMMIT);
	fprintf(fp, "energy: %e\n", sim->total_energy);
	fprintf(fp, "time: %e\n", sim->total_sim_time);
	fprintf(fp, "total_spikes: %ld\n", sim->total_spikes);
	fprintf(fp, "total_packets: %ld\n", sim->total_messages_sent);
	fprintf(fp, "total_neurons_fired: %ld\n", sim->total_neurons_fired);
	fprintf(fp, "wall_time: %lf\n", sim->wall_time);
	fprintf(fp, "timesteps: %ld\n", sim->timesteps);
}

void sim_spike_trace_write_header(const struct simulation *const sim)
{
	assert(sim->spike_trace_fp != NULL);
	fprintf(sim->spike_trace_fp, "gid.nid,timestep\n");

	return;
}

void sim_potential_trace_write_header(
	const struct simulation *const sim, const struct network *net)
{
	// Write csv header for probe outputs - record which neurons have been
	//  probed
	for (int i = 0; i < net->external_input_count; i++)
	{
		const struct input *in = &(net->external_inputs[i]);

		if (sim->potential_trace_fp && sim->log_potential)
		{
			fprintf(sim->potential_trace_fp, "i.%d,", in->id);
		}
	}

	for (int i = 0; i < net->neuron_group_count; i++)
	{
		const struct neuron_group *group = &(net->groups[i]);
		for (int j = 0; j < group->neuron_count; j++)
		{
			const struct neuron *n = &(group->neurons[j]);
			if (sim->potential_trace_fp && sim->log_potential &&
				n->log_potential)
			{
				fprintf(sim->potential_trace_fp, "%d.%d,",
					group->id, n->id);
			}
		}
	}

	if (sim->potential_trace_fp)
	{
		fputc('\n', sim->potential_trace_fp);
	}

	return;
}

void sim_message_trace_write_header(const struct simulation *const sim)
{
	assert(sim->message_trace_fp);
	fprintf(sim->message_trace_fp, "timestep,src_neuron,");
	fprintf(sim->message_trace_fp, "src_hw,dest_hw,hops,spikes,");
	fprintf(sim->message_trace_fp, "generation_latency,network_latency,");
	fprintf(sim->message_trace_fp, "processing_latency,blocking_latency,");
	fprintf(sim->message_trace_fp, "src_x,src_y,dest_x,dest_y\n");
}

void sim_trace_record_spikes(
	const struct simulation *const sim, const struct network *net)
{
	// A trace of all spikes that are generated
	int spike_probe_count = 0;
	assert(sim->spike_trace_fp != NULL);

	for (int i = 0; i < net->external_input_count; i++)
	{
		const struct input *in = &(net->external_inputs[i]);
		if (in->send_spike)
		{
			fprintf(sim->spike_trace_fp, "i.%d,%d,", in->id,
				in->send_spike);
		}
	}

	for (int i = 0; i < net->neuron_group_count; i++)
	{
		const struct neuron_group *group = &(net->groups[i]);

		for (int j = 0; j < group->neuron_count; j++)
		{
			const struct neuron *n = &(group->neurons[j]);
			if (n->log_spikes && n->fired)
			{
				fprintf(sim->spike_trace_fp, "%d.%d,%ld\n",
					n->group->id, n->id, sim->timesteps);
				spike_probe_count++;
			}
		}
	}

	return;
}

void sim_trace_record_potentials(
	const struct simulation *const sim, const struct network *net)
{
	// Each line of this csv file is the potential of all probed neurons for
	//  one time-step
	int potential_probe_count = 0;

	for (int i = 0; i < net->external_input_count; i++)
	{
		const struct input *in = &(net->external_inputs[i]);
		if (sim->potential_trace_fp)
		{
			fprintf(sim->potential_trace_fp, "%lf,", in->spike_val);
		}
	}

	for (int i = 0; i < net->neuron_group_count; i++)
	{
		const struct neuron_group *group = &(net->groups[i]);
		for (int j = 0; j < group->neuron_count; j++)
		{
			const struct neuron *n = &(group->neurons[j]);
			if (sim->potential_trace_fp && n->log_potential)
			{
				fprintf(sim->potential_trace_fp, "%lf,",
					n->potential);
				potential_probe_count++;
			}
		}
	}

	// Each timestep takes up a line in the respective csv file
	if (sim->potential_trace_fp && (potential_probe_count > 0))
	{
		fputc('\n', sim->potential_trace_fp);
	}

	return;
}

void sim_trace_record_message(
	const struct simulation *const sim, const struct message *const m)
{
	fprintf(sim->message_trace_fp, "%ld,", sim->timesteps);
	assert(m->src_neuron != NULL);
	fprintf(sim->message_trace_fp, "%d.%d,", m->src_neuron->group->id,
		m->src_neuron->id);
	assert(m->src_tile != NULL);
	assert(m->src_core != NULL);
	fprintf(sim->message_trace_fp, "%d.%d,", m->src_tile->id,
		m->src_core->id);

	assert(m->dest_tile != NULL);
	assert(m->dest_tile != NULL);
	fprintf(sim->message_trace_fp, "%d.%d,", m->dest_tile->id,
		m->dest_core->id);
	fprintf(sim->message_trace_fp, "%d,", m->hops);
	fprintf(sim->message_trace_fp, "%d,", m->spikes);
	fprintf(sim->message_trace_fp, "%le,", m->generation_latency);
	fprintf(sim->message_trace_fp, "%le,", m->network_latency);
	fprintf(sim->message_trace_fp, "%le,", m->receive_latency);
	fprintf(sim->message_trace_fp, "%le,", m->blocked_latency);

	// Print details about the route the message took
	fprintf(sim->message_trace_fp, "%d,", m->src_tile->x);
	fprintf(sim->message_trace_fp, "%d,", m->src_tile->y);
	fprintf(sim->message_trace_fp, "%d,", m->dest_tile->x);
	fprintf(sim->message_trace_fp, "%d\n", m->dest_tile->y);

	return;
}

int sim_poisson_input(const double firing_probability)
{
	// Simulate a single external input (as one neuron) for a timestep
	//  Return 1 if the input fires, 0 otherwise
	double rand_uniform;
	int input_fired;

	rand_uniform = (double) rand() / RAND_MAX;
	input_fired = rand_uniform < firing_probability;

	return input_fired;
}

int sim_rate_input(const double firing_rate, double *current)
{
	int input_fired;

	// Note: rate-based input (without randomization) is equivalent to a
	//  neuron with a fixed bias.
	TRACE2("rate input:%lf\n", firing_rate);
	*current += firing_rate;
	if (*current > 255.0)
	{
		*current = 0;
		input_fired = 1;
	}
	else
	{
		input_fired = 0;
	}

	TRACE2("input fired value:%d\n", input_fired);
	return input_fired;
}
