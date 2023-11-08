// Copyright (c) 2023 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
// arch.c
#include <ctype.h>
#include <stdlib.h>
#include <assert.h>

#include "print.h"
#include "arch.h"
#include "network.h"
#include "description.h"

struct architecture *arch_init(void)
{
	struct architecture *arch;

	arch = (struct architecture *) malloc(sizeof(struct architecture));
	if (arch == NULL)
	{
		printf("%ld\n", sizeof(struct core));
		printf("%ld\n", sizeof(struct tile));
		INFO("Error: Couldn't allocate %ld bytes.\n",
			sizeof(struct architecture));
		INFO("Error: Architecture couldn't be created.\n");
		exit(1);
	}

	arch->tile_count = 0;
	arch->core_count = 0;
	arch->time_barrier = 0.0;
	arch->is_init = 0;

	for (int i = 0; i < ARCH_MAX_TILES; i++)
	{
		// Initialize tile
		struct tile *t = &(arch->tiles[i]);

		t->energy = 0.0;
		t->time = 0.0;
		t->energy_east_west_hop = 0.0;
		t->time_east_west_hop = 0.0;
		t->energy_north_south_hop = 0.0;
		t->time_north_south_hop = 0.0;
		t->energy_spike_within_tile = 0.0;
		t->time_spike_within_tile = 0.0;
		t->blocked_until = 0.0;
		t->id = -1;
		t->x = -1;
		t->y = -1;
		t->core_count = 0;
		t->is_blocking = 0;
		t->max_dimensions = 0;
		t->width = 0;

		for (int j = 0; j < ARCH_MAX_CORES; j++)
		{
			// Initialize core
			struct core *c = &(t->cores[j]);
			c->t = NULL;
			c->next_timing = NULL;
			c->energy = 0.0;
			c->time = 0.0;
			c->blocked_until = 0.0;
			c->id = -1;
			c->buffer_pos = 0;
			c->is_blocking = 0;
			c->neuron_count = 0;
			c->curr_neuron = 0;
			c->neurons_left = 0;
			c->curr_axon = 0;

			c->axon_in.energy = 0.0;
			c->axon_in.time = 0.0;
			c->axon_in.packets_in = 0;
			c->axon_in.packet_size = 0;
			c->axon_in.packets_buffer = 0;
			c->axon_in.spikes_buffer = 0;
			c->axon_in.map_count = 0;

			c->synapse.spikes_buffer = 0;
			c->synapse.weights_per_word = 0;
			c->synapse.word_bits = 0;
			c->synapse.weight_bits = 0;
			c->synapse.spikes_processed = 0;
			c->synapse.memory_reads = 0;
			c->synapse.energy = 0.0;
			c->synapse.time = 0.0;
			c->synapse.energy_spike_op = 0.0;
			c->synapse.time_spike_op = 0.0;
			c->synapse.energy_memory_access = 0.0;
			c->synapse.time_memory_access = 0.0;

			c->dendrite.energy = 0.0;
			c->dendrite.time = 0.0;

			c->soma.energy = 0.0;
			c->soma.time = 0.0;
			c->soma.neurons_fired = 0;

			c->axon_out.energy = 0.0;
			c->axon_out.time = 0.0;

			for (int k = 0; i < ARCH_MAX_CONNECTION_MAP; i++)
			{
				// Initialize connection map
				c->axon_in.map[k].connection_count = 0;
				c->axon_in.map[k].active_synapses = 0;
				c->axon_in.map[k].spikes_received = 0;
				c->axon_in.map[k].receive_latency = 0.0;
				c->axon_out.map_ptr[k] = NULL;
				c->axon_in.map[k].last_updated = 0;
				c->axon_in.map[k].pre_neuron = NULL;
			}
		}
	}

	return arch;
}

void arch_free(struct architecture *const arch)
{
	for (int i = 0; i < arch->tile_count; i++)
	{
		struct tile *t = &(arch->tiles[i]);

		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);

			free(c->neurons);
			c->neurons = NULL;

			for (int k = 0; k < c->axon_in.map_count; k++)
			{
				struct connection_map *a = &(c->axon_in.map[k]);
				free(a->connections);
				a->connections = NULL;
			}
		}
	}
	free(arch);
}

int arch_create_noc(struct architecture *const arch, struct attributes *attr,
	const int attribute_count)
{
	int tile_id = 0;

	if (arch->tile_count <= 0)
	{
		// The NoC interconnect is built after tiles are all defined
		//  This is because we link the tiles together in the NoC mesh
		INFO("Error: NoC must be built after tiles defined.\n");
		exit(1);
	}

	// Default values
	arch->noc_dimensions = 2;
	arch->noc_width = -1;
	arch->noc_height = -1;

	for (int i = 0; i < attribute_count; i++)
	{
		struct attributes *a = &(attr[i]);

		if (strncmp("dimensions", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%d", &arch->noc_dimensions);
		}
		else if (strncmp("width", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%d", &arch->noc_width);
		}
		else if (strncmp("height", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%d", &arch->noc_height);
		}
	}
	assert((arch->noc_height * arch->noc_width) <= ARCH_MAX_TILES);

	for (int y = 0; y < arch->noc_height; y++)
	{
		for (int x = 0; x < arch->noc_width; x++)
		{
			struct tile *t = &(arch->tiles[tile_id]);
			int north_x, north_y, east_x, east_y, south_x, south_y;
			int west_x, west_y, link_count;

			tile_id++;
			t->x = x;
			t->y = y;

			north_x = t->x;
			north_y = t->y - 1;
			east_x = t->x + 1;
			east_y = t->y;
			south_x = t->x;
			south_y = t->y + 1;
			west_x = t->x - 1;
			west_y = t->y;

			link_count = 0;
			TRACE1("tid:%d (x:%d,y:%d)\n", t->id, t->x, t->y);
			if (north_y >= 0)
			{
				int lid = (north_y * arch->noc_width) + north_x;
				t->links[link_count] = &(arch->tiles[lid]);
				link_count++;
			}
			if (east_x < arch->noc_width)
			{
				int lid = (east_y * arch->noc_width) + east_x;
				t->links[link_count] = &(arch->tiles[lid]);
				link_count++;
			}
			if (south_y < arch->noc_height)
			{
				int lid = (south_y * arch->noc_width) + south_x;
				t->links[link_count] = &(arch->tiles[lid]);
				link_count++;
			}
			if (west_x >= 0)
			{
				int lid = (west_y * arch->noc_width) + west_x;
				t->links[link_count] = &(arch->tiles[lid]);
				link_count++;
			}
			assert(link_count > 0);
			assert(link_count <= 4);
			for (int i = 0; i < link_count; i++)
			{
				TRACE1("\tlink[%d]->%d\n", i,
					(t->links[i])->id);
			}
		}
	}

	arch->is_init = 1;
	TRACE1("NoC created, mesh, width:%d height:%d.\n", arch->noc_width,
		arch->noc_height);
	return 0;
}

int arch_create_tile(struct architecture *const arch, struct attributes *attr,
	const int attribute_count)
{
	struct tile *t;
	int id;

	if (arch->tile_count >= ARCH_MAX_TILES)
	{
		INFO("Error: Only %d tiles supported.\n", ARCH_MAX_TILES);
		exit(1);
	}

	id = arch->tile_count;
	arch->tile_count++;
	assert(arch->tile_count <= ARCH_MAX_TILES);
	t = &(arch->tiles[id]);

	t->id = id;
	t->energy = 0.0;
	t->time = 0.0;

	t->x = 0;
	t->y = 0;
	t->core_count = 0;
	for (int i = 0; i < ARCH_MAX_CORES; i++)
	{
		struct core *c = &(t->cores[i]);

		c->id = i;
		c->t = t;
	}

	// Set attributes
	t->is_blocking = 0;
	t->energy_spike_within_tile = 0.0;
	t->time_spike_within_tile = 0.0;
	t->energy_east_west_hop = 0.0;
	t->time_east_west_hop = 0.0;
	t->energy_north_south_hop = 0.0;
	t->time_north_south_hop = 0.0;

	for (int i = 0; i < attribute_count; i++)
	{
		struct attributes *a = &(attr[i]);

		if (strncmp("blocking", a->key, MAX_FIELD_LEN) == 0)
		{
			t->is_blocking = (strncmp(a->value_str, "True",
						  MAX_FIELD_LEN - 1) == 0);
		}
		else if (strncmp("energy_east_west", a->key, MAX_FIELD_LEN) ==
			0)
		{
			sscanf(a->value_str, "%lf", &t->energy_east_west_hop);
		}
		else if (strncmp("latency_east_west", a->key, MAX_FIELD_LEN) ==
			0)
		{
			sscanf(a->value_str, "%lf", &t->time_east_west_hop);
		}
		else if (strncmp("energy_north_south", a->key, MAX_FIELD_LEN) ==
			0)
		{
			sscanf(a->value_str, "%lf", &t->energy_north_south_hop);
		}
		else if (strncmp("latency_north_south", a->key,
				 MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf", &t->time_north_south_hop);
		}
		else if (strncmp("energy_spike_within_tile", a->key,
				 MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf",
				&t->energy_spike_within_tile);
		}
		else if (strncmp("latency_spike_within_tile", a->key,
				 MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf", &t->time_spike_within_tile);
		}
	}

	return t->id;
}

int arch_create_core(struct architecture *const arch, struct tile *const t,
	struct attributes *attr, const int attribute_count)
{
	struct core *c;
	unsigned int core_id;

	assert(t != NULL);
	core_id = t->core_count;
	t->core_count++;
	assert(t->core_count <= ARCH_MAX_CORES);

	c = &(t->cores[core_id]);
	c->offset = core_id;
	c->id = arch->core_count++;
	c->t = t;

	/*** Set attributes ***/
	c->is_blocking = 0;
	c->noise_type = NOISE_NONE;
	for (int i = 0; i < attribute_count; i++)
	{
		struct attributes *a = &(attr[i]);

		if (strncmp("blocking", a->key, MAX_FIELD_LEN) == 0)
		{
			c->is_blocking = (strncmp("True", a->value_str,
						  MAX_FIELD_LEN - 1) == 0);
		}
		if (strncmp("noise", a->key, MAX_FIELD_LEN) == 0)
		{
			c->noise_type = NOISE_FILE_STREAM;
			c->noise_stream = fopen(a->value_str, "r");
			INFO("Opening noise str: %s\n", a->value_str);
			if (c->noise_stream == NULL)
			{
				INFO("Error: Failed to open noise stream/\n");
				exit(1);
			}
		}
	}

	c->axon_in.energy = 0.0;
	c->axon_in.time = 0.0;
	c->axon_in.t = t;

	c->synapse.energy = 0.0;
	c->synapse.time = 0.0;

	c->dendrite.energy = 0.0;
	c->dendrite.time = 0.0;

	c->soma.energy = 0.0;
	c->soma.time = 0.0;

	c->axon_out.energy = 0.0;
	c->axon_out.time = 0.0;
	c->axon_out.t = t;

	c->neuron_count = 0;
	c->curr_neuron = 0;
	c->neurons = (struct neuron **) malloc(
		sizeof(struct neuron *) * ARCH_MAX_COMPARTMENTS);
	if (c->neurons == NULL)
	{
		INFO("Error: Couldn't allocate neuron memory.\n");
		exit(1);
	}
	for (int i = 0; i < 1024; i++)
	{
		c->neurons[i] = NULL;
	}

	c->energy = 0.0;
	c->time = 0.0;

	c->buffer_pos = BUFFER_SOMA;

	TRACE1("Core created id:%d (tile:%d).\n", c->id, t->id);
	return c->id;
}

void arch_create_axon_in(struct core *const c)
{
	struct axon_input *in;

	in = &(c->axon_in);
	in->energy = 0.0;
	in->time = 0.0;
	in->packet_size = 0;
	in->map_count = 0;
	// We already know a valid tile was given at this point
	in->t = c->t;

	TRACE2("Axon input created (c:%d.%d)\n", c->t->id, c->id);

	return;
}

void arch_create_synapse(struct core *const c,
	const struct attributes *const attr, const int attribute_count)
{
	struct synapse_processor *s;

	s = &(c->synapse);
	s->energy = 0.0;
	s->time = 0.0;

	/**** Set attributes ****/
	s->weight_bits = 8;
	s->word_bits = 64;
	s->energy_memory_access = 0.0;
	s->time_memory_access = 0.0;
	s->energy_spike_op = 0.0;
	s->time_spike_op = 0.0;
	for (int i = 0; i < attribute_count; i++)
	{
		const struct attributes *const curr = &(attr[i]);

		if (strncmp("weight_bits", curr->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(curr->value_str, "%d", &s->weight_bits);
		}
		if (strncmp("word_bits", curr->key, MAX_FIELD_LEN) == 0)
		{
			// The word size is the number of bits accessed with
			//  each memory read. The weight size is the number of
			//  bits for a single synaptic weight. A single memory
			//  read might return multiple weights
			sscanf(curr->value_str, "%d", &s->word_bits);
		}
		else if (strncmp("energy_memory", curr->key, MAX_FIELD_LEN) ==
			0)
		{
			sscanf(curr->value_str, "%lf",
				&s->energy_memory_access);
		}
		else if (strncmp("latency_memory", curr->key, MAX_FIELD_LEN) ==
			0)
		{
			sscanf(curr->value_str, "%lf", &s->time_memory_access);
		}
		else if (strncmp("energy_spike", curr->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(curr->value_str, "%lf", &s->energy_spike_op);
		}
		else if (strncmp("latency_spike", curr->key, MAX_FIELD_LEN) ==
			0)
		{
			sscanf(curr->value_str, "%lf", &s->time_spike_op);
		}
	}

	// Round up to the nearest word
	s->weights_per_word = s->word_bits / s->weight_bits;
	assert(s->weights_per_word > 0);

	TRACE1("Synapse processor created (c:%d.%d)\n", c->t->id, c->id);

	return;
}

void arch_create_soma(struct core *const c, struct attributes *attr,
	const int attribute_count)
{
	struct soma_processor *s;

	s = &(c->soma);
	s->energy = 0.0;
	s->time = 0.0;

	/*** Set attributes ***/
	s->model = NEURON_LIF;
	s->energy_active_neuron_update = 0.0;
	s->time_active_neuron_update = 0.0;
	s->energy_inactive_neuron_update = 0.0;
	s->time_inactive_neuron_update = 0.0;
	s->energy_spiking = 0.0;
	s->time_spiking = 0.0;
	s->leak_towards_zero = 1;
	for (int i = 0; i < attribute_count; i++)
	{
		struct attributes *a = &(attr[i]);

		if (strncmp("model", a->key, MAX_FIELD_LEN) == 0)
		{
			s->model = arch_parse_neuron_model(a->value_str);
		}
		else if (strncmp("energy_active", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf",
				&s->energy_active_neuron_update);
		}
		else if (strncmp("latency_active", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf",
				&s->time_active_neuron_update);
		}
		else if (strncmp("energy_inactive", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf",
				&s->energy_inactive_neuron_update);
		}
		else if (strncmp("latency_inactive", a->key, MAX_FIELD_LEN) ==
			0)
		{
			sscanf(a->value_str, "%lf",
				&s->time_inactive_neuron_update);
		}
		else if (strncmp("energy_spiking", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf", &s->energy_spiking);
		}
		else if (strncmp("latency_spiking", a->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(a->value_str, "%lf", &s->time_spiking);
		}
	}

	TRACE1("Soma processor created (c:%d.%d)\n", c->t->id, c->id);
	return;
}

void arch_create_axon_out(struct core *const c, struct attributes *attr,
	const int attribute_count)
{
	struct axon_output *out;

	out = &(c->axon_out);
	out->packets_out = 0;
	out->energy = 0.0;
	out->time = 0.0;

	/*** Set attributes ***/
	out->energy_access = 0.0;
	out->time_access = 0.0;
	for (int i = 0; i < attribute_count; i++)
	{
		struct attributes *curr = &(attr[i]);

		if (strncmp("energy", curr->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(curr->value_str, "%lf", &out->energy_access);
		}
		else if (strncmp("latency", curr->key, MAX_FIELD_LEN) == 0)
		{
			sscanf(curr->value_str, "%lf", &out->time_access);
		}
	}

	out->map_count = 0;
	// Track the tile the axon interfaces with
	out->t = c->t;
	TRACE1("Axon output created (c:%d.%d)\n", c->t->id, c->id);

	return;
}

void arch_create_connection_maps(struct architecture *const arch)
{
	TRACE1("Creating all connection maps.\n");
	//#pragma omp parallel for
	for (int i = 0; i < arch->tile_count; i++)
	{
		struct tile *t = &(arch->tiles[i]);
		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);
			for (int k = 0; k < c->neuron_count; k++)
			{
				arch_map_neuron_connections(c->neurons[k]);
			}
		}
	}

	TRACE1("Finished creating connection maps.\n");
	arch_print_connection_map_summary(arch);
}

void arch_print_connection_map_summary(struct architecture *const arch)
{
	int in_count, out_count, core_count, core_used;
	in_count = 0;
	out_count = 0;
	core_count = 0;

	INFO("** Mapping summary **\n");
	for (int i = 0; i < arch->tile_count; i++)
	{
		// For debug only, print the axon maps
		struct tile *t = &(arch->tiles[i]);
		for (int j = 0; j < t->core_count; j++)
		{
			struct core *c = &(t->cores[j]);

			core_used = 0;
			for (int k = 0; k < c->neuron_count; k++)
			{
#ifdef DEBUG
				struct neuron *n = c->neurons[k];
				TRACE2("\tnid:%d.%d ", n->group->id, n->id);
				TRACE2("i:%d o:%d\n", n->maps_in_count,
					n->maps_out_count);
#endif
				core_used = 1;
			}

			if (core_used)
			{
				INFO("cid:%d.%d n:%d i:%d o:%d\n", t->id,
					c->offset, c->neuron_count,
					c->axon_in.map_count,
					c->axon_out.map_count);
				in_count += c->axon_in.map_count;
				out_count += c->axon_out.map_count;
				core_count++;
			}
		}
	}
	INFO("Total cores: %d\n", core_count);
	INFO("Average in map count: %lf\n", (double) in_count / core_count);
	INFO("Average out map count: %lf\n", (double) out_count / core_count);

	return;
}

void arch_map_neuron_connections(struct neuron *const pre_neuron)
{
	// Setup the connections between neurons and map them to hardware
	int connection_count[ARCH_MAX_TILES * ARCH_MAX_CORES];
	struct core *cores[ARCH_MAX_TILES * ARCH_MAX_CORES];

	assert(pre_neuron->core != NULL);

	// Zero initialize all counters and tracking
	for (int x = 0; x < ARCH_MAX_TILES * ARCH_MAX_CORES; x++)
	{
		connection_count[x] = 0;
		cores[x] = NULL;
	}

	// Count how many connections go to each core from this neuron
	TRACE2("Counting connections for neuron nid:%d\n", pre_neuron->id);
	for (int conn = 0; conn < pre_neuron->connection_out_count; conn++)
	{
		TRACE2("Looking at connection id: %d\n", conn);
		struct connection *curr = &(pre_neuron->connections_out[conn]);
		struct core *dest_core = curr->post_neuron->core;
		int core_id = dest_core->id;

		connection_count[core_id]++;
		cores[core_id] = dest_core;
		TRACE2("Connected to dest core: %d\n", core_id);
	}

	TRACE2("Creating connections for neuron nid:%d\n", pre_neuron->id);
	int total_map_count = 0;
	for (int x = 0; x < ARCH_MAX_TILES * ARCH_MAX_CORES; x++)
	{
		if (connection_count[x] > 0)
		{
			// Create the connection map, and add it to both the
			//  destination and source cores
			arch_allocate_connection_map(
				pre_neuron, cores[x], connection_count[x]);
			total_map_count++;
		}
	}
	TRACE3("Counted all maps for nid:%d connection map count: %d\n",
		pre_neuron->id, total_map_count);
	assert(total_map_count < ARCH_MAX_CONNECTION_MAP);

	for (int conn = 0; conn < pre_neuron->connection_out_count; conn++)
	{
		// Add every connection to the map. Also link to the map in the
		//  post synaptic core / neuron
		struct connection *curr_connection =
			&(pre_neuron->connections_out[conn]);
		struct core *post_core = curr_connection->post_neuron->core;

		TRACE3("Adding connection:%d\n", conn);
		arch_add_connection_to_map(curr_connection, post_core);
	}
	TRACE2("Finished mapping connection to hardware for nid:%d.\n",
		pre_neuron->id);

	return;
}

void arch_allocate_connection_map(struct neuron *const pre_neuron,
	struct core *const post_core, const int connection_count)
{
	// For each connected core, create a new axon map at the destination
	//  core. Then link this axon to output of the source core. Finally
	//  update the presynaptic neuron and postsynaptic neuron to account for
	//  this
	assert(pre_neuron != NULL);
	assert(post_core != NULL);
	assert(connection_count >= 0);

	struct core *pre_core = pre_neuron->core;
	struct axon_input *axon_in = &(post_core->axon_in);
	int map_count = axon_in->map_count++;
	assert(axon_in->map_count >= 0);
	assert(axon_in->map_count < ARCH_MAX_CONNECTION_MAP);
	int map_size;

	TRACE2("axon in map count:%d for core:%d.%d, adding %d connections\n",
		map_count, post_core->id, post_core->t->id,
		connection_count);
	struct connection_map *map = &(axon_in->map[map_count]);

	TRACE3("Adding connection to core.\n");
	// Allocate the map and its connections at the post-synaptic (dest)
	//  core
	map_size = connection_count * sizeof(struct connection);
	TRACE3("Axon has %d connections, allocate %d bytes\n",
		connection_count, map_size);
	map->connections = malloc(connection_count * map_size);
	if (map->connections == NULL)
	{
		INFO("Error: Couldn't allocate map memory.\n");
		exit(1);
	}

	// Link to this map in the pre-synaptic (src) core
	map_count = pre_core->axon_out.map_count++;
	assert(pre_core->axon_out.map_count >= 0);
	assert(pre_core->axon_out.map_count < ARCH_MAX_CONNECTION_MAP);
	pre_core->axon_out.map_ptr[map_count] = map;
	if (pre_neuron->maps_out == NULL)
	{
		TRACE2("Setting neuron nid:%d axon out.\n", pre_neuron->id);
		pre_neuron->maps_out = &(pre_core->axon_out.map_ptr[map_count]);
		assert(pre_neuron->maps_out != NULL);
		assert(pre_neuron->maps_out[0] != NULL);
	}
	pre_neuron->maps_out_count++;
	assert(pre_neuron->maps_out_count >= 0);
	TRACE2("nid:%d.%d cid:%d.%d added one output axon, "
	       "axon out map_count:%d, neuron out map count:%d.\n",
		pre_neuron->group->id, pre_neuron->id, pre_core->t->id,
		pre_core->id, pre_core->axon_out.map_count,
		pre_neuron->maps_out_count);

	return;
}

void arch_add_connection_to_map(
	struct connection *const con, struct core *const post_core)
{
	// Add a given connection to the connection map in the post-synaptic
	//  (destination) core. Check to see if we already have a map for this
	//  source / destination core combination - if so we can reuse and add
	//  to that connection map. Otherwise, we need to use a new map.
	const int map_count = post_core->axon_in.map_count;

	assert(map_count > 0);
	assert(map_count <= ARCH_MAX_CONNECTION_MAP);
	TRACE3("Adding to connection to map:%d\n", map_count - 1);

	// Access the most recently created axon for the core
	struct connection_map *axon = &(post_core->axon_in.map[map_count - 1]);
	axon->connections[axon->connection_count++] = con;
	axon->pre_neuron = con->pre_neuron;

	// Update the post synaptic neuron to track
	if (con->post_neuron->maps_in == NULL)
	{
		// Point to the first mapping
		con->post_neuron->maps_in = axon;
	}

	// We might add a bunch of connections from another core coming into
	//  this one, then we need to update and track
	con->post_neuron->maps_in_count++;

	return;
}

int arch_parse_neuron_model(char *model_str)
{
	int model;

	if (strcmp(model_str, "leaky_integrate_fire") == 0)
	{
		model = NEURON_LIF;
	}
	else if (strcmp(model_str, "truenorth") == 0)
	{
		model = NEURON_TRUENORTH;
	}
	else
	{
		INFO("Error: No neuron model specified (%s)\n", model_str);
		exit(1);
	}

	return model;
}
