// Copyright (c) 2024 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
// models.hpp
#ifndef MODELS_HEADER_INCLUDED_
#define MODELS_HEADER_INCLUDED_

#include <map>
#include <memory> // For shared_ptr<T>
#include <optional>
#include <vector>

namespace sanafe
{
struct Synapse;
enum NeuronStatus: int { IDLE, UPDATED, FIRED};
enum NeuronResetModes
{
    NEURON_NO_RESET,
    NEURON_RESET_SOFT,
    NEURON_RESET_HARD,
    NEURON_RESET_SATURATE,
    NEURON_RESET_MODE_COUNT,
};

class SynapseModel
{
public:
    SynapseModel() {}
    virtual ~SynapseModel() {}
    virtual double update(std::optional<int> synapse_address) = 0;
    // Set synapse attributes
    virtual void set_attributes(const std::map<std::string, std::string> &attr) = 0;
};

class DendriteModel
{
public:
    DendriteModel() {}
    virtual ~DendriteModel() {}
    // TODO: figure out how this model works... do we update each compartment?
    //virtual void input(const double current_in, const int compartment) = 0;
    virtual double update(const std::vector<Synapse> &synapses_in) = 0;
    // Set global dendritic attributes
    virtual void set_attributes(const std::map<std::string, std::string> &attr) = 0;
    // Set per-compartment attributes
    virtual void set_attributes(const size_t compartment_id, const std::map<std::string, std::string> &attr) {}
    // Set per-branch attributes (between compartments)
    virtual void set_attributes(const size_t src_compartment_id, const size_t dest_compartment_id, const std::map<std::string, std::string> &attr) {}
};

class CurrentBasedSynapseModel: public SynapseModel
{
public:
    CurrentBasedSynapseModel();
    ~CurrentBasedSynapseModel() {}
    virtual double update(std::optional<int> synapse_address=std::nullopt);
    virtual void set_attributes(const std::map<std::string, std::string> &attr);

private:
    double weight, min_synaptic_resolution, current, synaptic_current_decay;
    int weight_bits;
};

class SingleCompartmentModel: public DendriteModel
{
public:
    SingleCompartmentModel();
    ~SingleCompartmentModel() {}
    virtual double update(const std::vector<Synapse> &synapses_in);
    virtual void set_attributes(const std::map<std::string, std::string> &attr);
private:
    double accumulated_charge, leak_decay;
};

class MultiTapModel: public DendriteModel
{
public:
    MultiTapModel();
    virtual double update(const std::vector<Synapse> &synapses_in);
    virtual void set_attributes(const std::map<std::string, std::string> &attr);
    virtual void set_attributes(const size_t compartment_id, const std::map<std::string, std::string> &attr);
    virtual void set_attributes(const size_t src_compartment_id, const size_t dest_compartment_id, const std::map<std::string, std::string> &attr);
private:
    // Use a simple dense matrix representation of tap to tap weights,
    //  assuming that the number of taps is (reasonably) small. For larger
    //  tap counts, a model implementing a sparse per-branch representation
    //  would be more space efficient
    std::vector<double> tap_voltages, next_voltages, weights;
};

class SomaModel
{
public:
    SomaModel(const int gid, const int nid) : group_id(gid), neuron_id(nid) {}
    virtual ~SomaModel() {}
    virtual NeuronStatus update(const std::optional<double> synapses_in=std::nullopt) = 0;
    virtual void set_attributes(const std::map<std::string, std::string> &attr) = 0;
    virtual double get_potential() { return 0.0; }
protected:
    const int group_id, neuron_id;
    // TODO: this might be useful context
    //const int mapped_tile_id, mapped_core_id, mapped_core_offset;
};

class LoihiLifModel: public SomaModel
{
public:
    LoihiLifModel(const int gid, const int nid);
    ~LoihiLifModel() {}
    void set_attributes(const std::map<std::string, std::string> &attr);
    NeuronStatus update(const std::optional<double> current_in);
    double get_potential() { return potential; }
private:
    bool force_update;
    int reset_mode, reverse_reset_mode;
    //int noise_type;
    double potential, leak_decay, bias, threshold, reverse_threshold;
        double reset, reverse_reset;
};

class TrueNorthModel: public SomaModel
{
public:
    TrueNorthModel(const int gid, const int nid);
    ~TrueNorthModel() {}
    void set_attributes(const std::map<std::string, std::string> &attr);
    NeuronStatus update(const std::optional<double> current_in=std::nullopt);
    double get_potential() { return potential; }
private:
    bool force_update;
    unsigned int random_range_mask;
    int reset_mode, reverse_reset_mode;
    bool leak_towards_zero;
    //int noise_type;
    double potential, leak, bias, threshold, reverse_threshold;
    double reset, reverse_reset;
};

NeuronResetModes model_parse_reset_mode(const std::string &str);
std::shared_ptr<SynapseModel> model_get_synapse(const std::string &model_name);
std::shared_ptr<DendriteModel> model_get_dendrite(const std::string &model_name);
std::shared_ptr<SomaModel> model_get_soma(const std::string &model_name, const int group_id, const int id);

}

#endif
