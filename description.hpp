#ifndef DESCRIPTION_HEADER_INCLUDED_
#define DESCRIPTION_HEADER_INCLUDED_

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sanafe
{
enum DescriptionRet
{
    RET_FAIL = -1,
    RET_OK = 0, // Anything >= 0 means successfully parsed
};

// Forward struct declarations
class Simulation;
class Architecture;
class Network;
class NeuronGroup;
class Neuron;
class Tile;
class Core;
struct CorePipelineConfiguration;
struct TilePowerMetrics;
struct NetworkOnChipConfiguration;

Architecture description_parse_arch_file(std::ifstream &fp);
int description_parse_net_file(std::ifstream &fp, Network &net, Architecture &arch);
void description_get_fields(std::vector<std::string_view> &fields, const std::string &line);
void description_read_network_entry(const std::vector<std::string_view> &fields, Architecture &arch, Network &net, const int line_number);
void parse_neuron_field(const std::string_view &neuron_field, size_t &group_id, size_t &neuron_id);
void parse_core_field(const std::string_view &core_field, size_t &tile_id, size_t &core_offset);
void parse_edge_field(const std::string_view &edge_field, size_t &group_id, size_t &neuron_id, size_t &dest_group_id, size_t &dest_neuron_id);
void parse_mapping_field(const std::string_view &mapping_field, size_t &group_id, size_t &neuron_id, size_t &tile_id, size_t &core_offset);
size_t field_to_int(const std::string_view &field);

Architecture description_parse_arch_section(const YAML::Node &arch_node);
void description_parse_tile_section(const YAML::Node &tile_node, Architecture &arch);
void description_parse_core_section(const YAML::Node &core_node, const size_t parent_tile_id, Architecture &arch);
void description_parse_axon_in_section(const YAML::Node &axon_in_node, Core &parent_core);
void description_parse_synapse_section(const YAML::Node &synapse_node, Core &parent_core);
void description_parse_dendrite_section(const YAML::Node &dendrite_node, Core &parent_core);
void description_parse_soma_section(const YAML::Node &soma_node, Core &parent_core);
void description_parse_axon_out_section(const YAML::Node &axon_out_node, Core &parent_core);
std::pair<size_t, size_t> description_parse_range(const std::string &tile_name);
CorePipelineConfiguration description_parse_core_pipeline(const YAML::Node &attributes);
TilePowerMetrics description_parse_tile_metrics(const YAML::Node &attributes);
NetworkOnChipConfiguration description_parse_noc_configuration(const YAML::Node &arch_node);
}

#endif
