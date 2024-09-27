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
#include <random>
#include <string>
#include <vector>

#include "arch.hpp"
#include "print.hpp"

namespace sanafe
{

// Forward declarations
struct Synapse;
enum NeuronStatus : int;

constexpr int default_weight_bits = 8; // Based on real-world H/W e.g., Loihi
constexpr int loihi_max_compartments{1024};

class CurrentBasedSynapseModel : public SynapseUnit
{
public:
    CurrentBasedSynapseModel() = default;
    CurrentBasedSynapseModel(const CurrentBasedSynapseModel &copy) = default;
    CurrentBasedSynapseModel(CurrentBasedSynapseModel &&other) = default;
    ~CurrentBasedSynapseModel() override = default;
    CurrentBasedSynapseModel &operator=(const CurrentBasedSynapseModel &other) = default;
    CurrentBasedSynapseModel &operator=(CurrentBasedSynapseModel &&other) = default;

    SynapseResult update(size_t synapse_address, bool read) override;
    void set_attribute(size_t synapse_address, const std::string &param_name, const ModelParam &param) override;

private:
    std::vector<double> weights{};
    double min_synaptic_resolution{0.0};
    int weight_bits{default_weight_bits};
};

class AccumulatorModel : public DendriteUnit
{
public:
    AccumulatorModel() = default;
    AccumulatorModel(const AccumulatorModel &copy) = default;
    AccumulatorModel(AccumulatorModel &&other) = default;
    ~AccumulatorModel() override = default;
    AccumulatorModel &operator=(const AccumulatorModel &other) = default;
    AccumulatorModel &operator=(AccumulatorModel &&other) = default;

    DendriteResult update(size_t neuron_address, std::optional<Synapse> synapse_in) override;
    void set_attribute(size_t neuron_address, const std::string &param_name, const ModelParam &param) override;

private:
    std::vector<double> accumulated_charges{std::vector<double>(loihi_max_compartments, 0.0)};
    std::vector<long int> timesteps_simulated{std::vector<long int>(loihi_max_compartments, 0)};
    double leak_decay{0.0};
};

class MultiTapModel1D : public DendriteUnit
{
public:
    MultiTapModel1D() = default;
    MultiTapModel1D(const MultiTapModel1D &copy) = default;
    MultiTapModel1D(MultiTapModel1D &&other) = default;
    ~MultiTapModel1D() override = default;
    MultiTapModel1D &operator=(const MultiTapModel1D &other) = default;
    MultiTapModel1D &operator=(MultiTapModel1D &&other) = default;

    DendriteResult update(size_t neuron_address, std::optional<Synapse> synapse_in) override;
    void set_attribute(size_t neuron_address, const std::string &param_name, const ModelParam &param) override;

private:
    // Modeling a 1D dendrite with taps
    std::vector<double> tap_voltages{std::vector<double>(1, 0.0)};
    std::vector<double> next_voltages{std::vector<double>(1, 0.0)};
    std::vector<double> space_constants{std::vector<double>(0)};
    std::vector<double> time_constants{std::vector<double>(1, 0.0)};
    long int timesteps_simulated{0L};
};

class LoihiLifModel : public SomaUnit
{
public:
    LoihiLifModel() = default;
    LoihiLifModel(const LoihiLifModel &copy) = default;
    LoihiLifModel(LoihiLifModel &&other) = default;
    ~LoihiLifModel() override = default;
    LoihiLifModel &operator=(const LoihiLifModel &other) = delete;
    LoihiLifModel &operator=(LoihiLifModel &&other) = delete;

    void set_attribute(size_t neuron_address, const std::string &param_name, const ModelParam &param) override;
    SomaResult update(size_t neuron_address, std::optional<double> current_in) override;
    double get_potential(const size_t neuron_address) override
    {
        return compartments[neuron_address].potential;
    }

    struct LoihiCompartment
    {
        bool force_update{false};
        long int timesteps_simulated{0L};
        int reset_mode{NEURON_RESET_HARD};
        int reverse_reset_mode{NEURON_NO_RESET};
        double potential{0.0};
        double leak_decay{1.0};
        double bias{0.0};
        double threshold{0.0};
        double reverse_threshold{0.0};
        double reset{0.0};
        double reverse_reset{0.0};
    };

private:
    std::vector<LoihiCompartment> compartments{1024};
};

constexpr int truenorth_max_neurons{4096};
class TrueNorthModel : public SomaUnit
{
public:
    TrueNorthModel() = default;
    TrueNorthModel(const TrueNorthModel &copy) = default;
    TrueNorthModel(TrueNorthModel &&other) = default;
    ~TrueNorthModel() override = default;
    TrueNorthModel &operator=(const TrueNorthModel &other) = delete;
    TrueNorthModel &operator=(TrueNorthModel &&other) = delete;

    void set_attribute(const size_t neuron_address, const std::string &param_name, const ModelParam &param) override;
    SomaResult update(const size_t neuron_address, std::optional<double> current_in = std::nullopt) override;
    double get_potential(const size_t neuron_address) override
    {
        return neurons[neuron_address].potential;
    }

    struct TrueNorthNeuron
    {
        bool force_update{false};
        unsigned int random_range_mask{0U};
        int reset_mode{NEURON_RESET_HARD};
        int reverse_reset_mode{NEURON_NO_RESET};
        bool leak_towards_zero{true};
        // int noise_type;
        double potential{0.0};
        double leak{0.0}; // Default is no leak (potential decay)
        double bias{0.0};
        double threshold{0.0};
        double reverse_threshold{0.0};
        double reset{0.0};
        double reverse_reset{0.0};
    };

private:
    std::vector<TrueNorthNeuron> neurons{truenorth_max_neurons};
};

class InputModel : public SomaUnit
{
public:
    InputModel() = default;
    InputModel(const InputModel &copy) = default;
    InputModel(InputModel &&other) = default;
    ~InputModel() override = default;
    InputModel &operator=(const InputModel &other) = delete;
    InputModel &operator=(InputModel &&other) = delete;

    void set_attribute(size_t neuron_address, const std::string &param_name, const ModelParam &param) override;
    SomaResult update(size_t neuron_address, std::optional<double> current_in = std::nullopt) override;

private:
    std::vector<bool> spikes{};
    std::vector<bool>::const_iterator curr_spike{spikes.begin()};
    std::uniform_real_distribution<double> uniform_distribution{0.0, 1.0};
    std::random_device rd;
    std::mt19937 gen{rd()};
    double poisson_probability{0.0};
    bool send_spike{false};
};

std::shared_ptr<SynapseUnit> model_get_synapse(const std::string &model_name);
std::shared_ptr<DendriteUnit> model_get_dendrite(const std::string &model_name);
std::shared_ptr<SomaUnit> model_get_soma(const std::string &model_name);
NeuronResetModes model_parse_reset_mode(const std::string &str);
}

#endif
