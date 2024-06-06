#include <cassert>
#include <cmath>
#include <optional>
#include <sstream>
#include <vector>

#include "description.hpp"
#include "models.hpp"
#include "plugins.hpp"
#include "print.hpp"

// *** Synapse models ***
sanafe::CurrentBasedSynapseModel::CurrentBasedSynapseModel()
{
    current = 0.0;
    min_synaptic_resolution = 0.0;
}

double sanafe::CurrentBasedSynapseModel::update(
        const std::optional<int> synapse_address)
{

    current *= synaptic_current_decay;
    if (fabs(current) < min_synaptic_resolution)
    {
        current = 0.0;
    }
    if (synapse_address.has_value())
    {
        current += weight;
    }

    return current;
}

void sanafe::CurrentBasedSynapseModel::set_attributes(
        const std::map<std::string, std::string> &attr)
{
    weight = 0.0;
    weight_bits = 8;
    synaptic_current_decay = 0.0;
    for (const auto &a : attr)
    {
        const std::string &key = a.first;
        const std::string &value_str = a.second;
        std::istringstream ss(value_str);

        if ((key[0] == 'w') || (key == "weight"))
        {
            ss >> weight;
        }
    }

    min_synaptic_resolution = (1.0 / weight_bits);
}

// *** Dendrite models ***
sanafe::SingleCompartmentModel::SingleCompartmentModel()
{
    accumulated_charge = 0.0;
    leak_decay = 0.0;
}

double sanafe::SingleCompartmentModel::update(
        const std::vector<Synapse> synapses_in)
{
    accumulated_charge *= leak_decay;
    //INFO("accumulated:%e\n", accumulated_charge);
    for (const auto &synapse : synapses_in)
    {
        // Integrate any input current
        accumulated_charge += synapse.current;
    }

    return accumulated_charge;
}

void sanafe::SingleCompartmentModel::set_attributes(
        const std::map<std::string, std::string> &attr)
{
    for (const auto &a : attr)
    {
        const std::string &key = a.first;
        const std::string &value_str = a.second;
        std::istringstream ss(value_str);

        if (key == "dendrite_leak_decay")
        {
            ss >> leak_decay;
        }
    }
}

sanafe::MultiTapModel::MultiTapModel()
{
    const int default_taps = 1;
    tap_voltages = std::vector<double>(default_taps, 0.0);
    next_voltages = std::vector<double>(default_taps, 0.0);
    weights = std::vector<double>(default_taps * default_taps, 0.0);
}

double sanafe::MultiTapModel::update(const std::vector<Synapse> &synapses_in)
{

    const size_t taps = tap_voltages.size();
    // Avoid reallocating this temporary buffer every time
    for (size_t t = 0; t < taps; t++)
    {
        next_voltages[t] = 0.0;
    }
    for (size_t src = 0; src < taps; src++)
    {
        for (size_t dest = 0; dest < taps; dest++)
        {
            const size_t taps = tap_voltages.size();
            const size_t weight_idx = (src * taps) + dest;
            next_voltages[dest] += tap_voltages[src] * weights[weight_idx];
        }
        INFO("Tap %lu potential:%lf\n", src, tap_voltages[src]);
        //printf("%lf\t", tap_voltages[src]);
    }
    //printf("\n");
    for (size_t t = 0; t < taps; t++)
    {
        tap_voltages[t] = next_voltages[t];
    }
    for (const auto &synapse : synapses_in)
    {
        assert(synapse.compartment >= 0);
        assert((size_t) synapse.compartment < tap_voltages.size());
        tap_voltages[synapse.compartment] += synpase_in.current;
    }

    // Return current for most proximal tap (which is always the first tap)
    return tap_voltages[0];
}

void sanafe::MultiTapModel::set_attributes(
        const std::map<std::string, std::string> &attr)
{
    for (const auto &a : attr)
    {
        const std::string &key = a.first;
        const std::string &value_str = a.second;
        std::istringstream ss(value_str);

        if (key == "taps")
        {
            int taps;
            ss >> taps;
            if (taps <= 0)
            {
                throw(std::invalid_argument("Error: Invalid tap count."));
            }
            tap_voltages = std::vector<double>(taps, 0.0);
            next_voltages = std::vector<double>(taps, 0.0);
            weights = std::vector<double>(taps * taps, 0.0);
        }
    }
}

void sanafe::MultiTapModel::set_attributes(const size_t compartment_id,
        const std::map<std::string, std::string> &attr)
{
    const size_t taps = tap_voltages.size();
    if (compartment_id >= taps)
    {
        throw std::invalid_argument("Error: Compartment out of range");
    }
    for (const auto &a : attr)
    {
        const std::string &key = a.first;
        const std::string &value_str = a.second;
        std::istringstream ss(value_str);

        if (key == "time_constant")
        {
            double time_constant;
            ss >> time_constant;
            const size_t taps = tap_voltages.size();
            const size_t idx = (taps * compartment_id) + compartment_id;
            weights[idx] = time_constant;
        }
    }
}

void sanafe::MultiTapModel::set_attributes(const size_t src_compartment_id,
        const size_t dest_compartment_id,
        const std::map<std::string, std::string> &attr)
{
    const size_t taps = tap_voltages.size();
    if (src_compartment_id >= taps)
    {
        throw std::invalid_argument("Error: src compartment out of range");
    }
    if (dest_compartment_id >= taps)
    {
        throw std::invalid_argument("Error: dest compartment out of range");
    }

    for (const auto &a : attr)
    {
        const std::string &key = a.first;
        const std::string &value_str = a.second;
        std::istringstream ss(value_str);

        if (key == "space_constant")
        {
            double space_constant;
            ss >> space_constant;
            const size_t taps = tap_voltages.size();
            const size_t idx =
                    (taps * src_compartment_id) + dest_compartment_id;
            weights[idx] = space_constant;
            // Update the src compartment, subtracting some of its
            //  potential which is added to the destination
            //  compartment. Formulate this for the matrix mul
            const size_t src_identity =
                    (src_compartment_id * taps) + src_compartment_id;
            weights[src_identity] -= space_constant;
        }
    }
}

// **** Soma models ****
sanafe::LoihiLifModel::LoihiLifModel(const int gid, const int nid)
        : sanafe::SomaModel(gid, nid)
{
    force_update = false;
    potential = 0.0;
    bias = 0.0;
    threshold = 0.0;
    reverse_threshold = 0.0;
    reset = 0.0;
    reverse_reset = 0.0;
    reset_mode = sanafe::NEURON_RESET_HARD;
    reverse_reset_mode = sanafe::NEURON_NO_RESET;
    // Default is no leak (potential decay), i.e., the potential for the
    //  next timestep is 100% of the previous timestep's
    leak_decay = 1.0;

    return;
}

void sanafe::LoihiLifModel::set_attributes(
        const std::map<std::string, std::string> &attr)
{
    for (auto a : attr)
    {
        const std::string &key = a.first;
        const std::string &value_str = a.second;
        std::istringstream ss(value_str);
        if (key == "threshold")
        {
            ss >> threshold;
        }
        else if (key == "reverse_threshold")
        {
            ss >> reverse_threshold;
        }
        else if (key == "reset")
        {
            ss >> reset;
        }
        else if (key == "reverse_reset")
        {
            ss >> reverse_reset;
        }
        else if (key == "reset_mode")
        {
            reset_mode = model_parse_reset_mode(value_str);
        }
        else if (key == "reverse_reset_mode")
        {
            reverse_reset_mode = model_parse_reset_mode(value_str);
        }
        else if (key == "leak_decay")
        {
            ss >> leak_decay;
        }
        else if (key == "bias")
        {
            ss >> bias;
        }
        else if (key == "force_update")
        {
            ss >> force_update;
        }
    }
}

sanafe::NeuronStatus sanafe::LoihiLifModel::update(
        const std::optional<double> current_in)
{
    // Calculate the change in potential since the last update e.g.
    //  integate inputs and apply any potential leak
    TRACE1("Updating potential, before:%f\n", potential);
    sanafe::NeuronStatus state = sanafe::IDLE;
    // Update soma, if there are any received spikes, there is a non-zero
    //  bias or we force the neuron to update every time-step
    if ((std::fabs(potential) > 0.0) || current_in.has_value() ||
            (std::fabs(bias) > 0.0) || force_update)
    {
        // Neuron is turned on and potential write
        state = sanafe::UPDATED;
    }


    potential *= leak_decay;
    // Add randomized noise to potential if enabled
    /*
    if (noise_type == NOISE_FILE_STREAM)
    {
        // TODO: fix noise generation. This depends on which core
        //  is simulating the neuron. So somehow the neuron can still
        //  need information about which core it is executing on
        double random_potential = sim_generate_noise(n);
        n->potential += random_potential;
    }
    */
    // Add the synaptic / dendrite current to the potential
    TRACE1("bias:%lf potential before:%lf current_in:%lf\n", bias,
            potential, current_in.value());
    potential += bias;

    if (current_in.has_value())
    {
        potential += current_in.value();
    }
    TRACE1("leak decay:%lf bias:%lf threshold:%lf potential after:%lf\n",
            leak_decay, bias, threshold, potential);
    TRACE1("Updating potential, after:%f\n", potential);

    // Check against threshold potential (for spiking)
    if (potential > threshold)
    {
        if (reset_mode == sanafe::NEURON_RESET_HARD)
        {
            potential = reset;
        }
        else if (reset_mode == sanafe::NEURON_RESET_SOFT)
        {
            potential -= threshold;
        }
        state = sanafe::FIRED;
        TRACE1("Neuron fired\n");
    }
    // Check against reverse threshold
    if (potential < reverse_threshold)
    {
        if (reverse_reset_mode == sanafe::NEURON_RESET_SOFT)
        {
            potential -= reverse_threshold;
        }
        else if (reverse_reset_mode == sanafe::NEURON_RESET_HARD)
        {
            potential = reverse_reset;
        }
        else if (reverse_reset_mode == sanafe::NEURON_RESET_SATURATE)
        {
            potential = reverse_threshold;
        }
    }
    return state;
}

sanafe::TrueNorthModel::TrueNorthModel(const int gid, const int nid)
        : sanafe::SomaModel(gid, nid)
{
    force_update = false;
    potential = 0.0;
    bias = 0.0;
    threshold = 0.0;
    reverse_threshold = 0.0;
    reset = 0.0;
    reverse_reset = 0.0;
    reset_mode = sanafe::NEURON_RESET_HARD;
    reverse_reset_mode = sanafe::NEURON_NO_RESET;
    leak_towards_zero = true;
    random_range_mask = 0U;
    // Default is no leak (potential decay), i.e., the potential for the
    //  next timestep is 100% of the previous timestep's
    leak = 0.0;
}

void sanafe::TrueNorthModel::set_attributes(
        const std::map<std::string, std::string> &attr)
{
    for (auto a : attr)
    {
        const std::string &key = a.first;
        const std::string &value_str = a.second;
        std::istringstream ss(value_str);
        if (key == "threshold")
        {
            ss >> threshold;
        }
        else if (key == "reverse_threshold")
        {
            ss >> reverse_threshold;
        }
        else if (key == "reset")
        {
            ss >> reset;
        }
        else if (key == "reverse_reset")
        {
            ss >> reverse_reset;
        }
        else if (key == "reset_mode")
        {
            reset_mode = model_parse_reset_mode(value_str);
        }
        else if (key == "reverse_reset_mode")
        {
            reverse_reset_mode = model_parse_reset_mode(value_str);
        }
        else if (key == "leak")
        {
            ss >> leak;
        }
        else if (key == "bias")
        {
            ss >> bias;
        }
        else if (key == "force_update")
        {
            ss >> force_update;
        }
        else if (key == "leak_towards_zero")
        {
            ss >> leak_towards_zero;
        }
    }
}

sanafe::NeuronStatus sanafe::TrueNorthModel::update(
        const std::optional<double> current_in)
{
    bool randomize_threshold;
    sanafe::NeuronStatus state = sanafe::IDLE;

    if ((std::fabs(potential) > 0.0) || current_in.has_value() ||
            (std::fabs(bias) > 0.0) || force_update)
    {
        // Neuron is turned on and potential write
        state = sanafe::UPDATED;
    }

    // Apply leak
    if (leak_towards_zero)
    {
        // TODO: what happens if we're above zero but by less
        //  than the leak amount (for convergent), will we
        //  oscillate between the two? Does it matter
        if (potential > 0.0)
        {
            // TrueNorth uses additive leak
            potential -= leak;
        }
        else if (potential < 0.0)
        {
            potential += leak;
        }
        // else equals zero, so no leak is applied
    }
    else
    {
        potential += leak;
    }

    potential += bias;
    if (current_in.has_value())
    {
        potential += current_in.value();
    }
    // Apply thresholding and reset
    double v = potential;
    randomize_threshold = (random_range_mask != 0);
    if (randomize_threshold)
    {
        unsigned int r = rand() & random_range_mask;
        v += static_cast<double>(r);
    }
    TRACE2("v:%lf +vth:%lf mode:%d -vth:%lf mode:%d\n", v, threshold,
            reset_mode, reverse_threshold, reverse_reset_mode);
    if (v >= threshold)
    {
        if (reset_mode == NEURON_RESET_HARD)
        {
            potential = reset;
        }
        else if (reset_mode == NEURON_RESET_SOFT)
        {
            potential -= threshold;
        }
        else if (reset_mode == NEURON_RESET_SATURATE)
        {
            potential = threshold;
        }
        state = sanafe::FIRED;
    }
    else if (v <= reverse_threshold)
    {
        if (reverse_reset_mode == NEURON_RESET_HARD)
        {
            potential = reverse_reset;
        }
        else if (reverse_reset_mode == NEURON_RESET_SOFT)
        {
            potential += reverse_threshold;
        }
        else if (reverse_reset_mode == NEURON_RESET_SATURATE)
        {
            potential = reverse_threshold;
        }
        // No spike is generated
    }
    TRACE2("potential:%lf threshold %lf\n", potential, threshold);
    return state;
}

sanafe::NeuronResetModes sanafe::model_parse_reset_mode(const std::string &str)
{
    sanafe::NeuronResetModes reset_mode;

    if (str == "none")
    {
        reset_mode = sanafe::NEURON_NO_RESET;
    }
    else if (str == "soft")
    {
        reset_mode = sanafe::NEURON_RESET_SOFT;
    }
    else if (str == "hard")
    {
        reset_mode = sanafe::NEURON_RESET_HARD;
    }
    else if (str == "saturate")
    {
        reset_mode = sanafe::NEURON_RESET_SATURATE;
    }
    else
    {
        throw std::invalid_argument("Error: reset mode not recognized");
    }

    return reset_mode;
}

std::shared_ptr<sanafe::SynapseModel> sanafe::model_get_synapse(
        const std::string &model_name)
{
    if (model_name == "cuba")
    {
        return std::shared_ptr<SynapseModel>(new CurrentBasedSynapseModel());
    }
    else
    {
        throw std::invalid_argument("Synapse model not supported.");
    }
}

std::shared_ptr<sanafe::DendriteModel> sanafe::model_get_dendrite(
        const std::string &model_name)
{
    if (model_name == "single_compartment")
    {
        return std::shared_ptr<DendriteModel>(new SingleCompartmentModel());
    }
    else if (model_name == "taps")
    {
        return std::shared_ptr<DendriteModel>(new MultiTapModel());
    }
    else
    {
        throw std::invalid_argument("Model not supported.");
    }
}

std::shared_ptr<sanafe::SomaModel> sanafe::model_get_soma(
        const std::string &model_name, const int group_id, const int id)
{
    if (model_name == "leaky_integrate_fire")
    {
        return std::shared_ptr<SomaModel>(new LoihiLifModel(group_id, id));
    }
    else if (model_name == "truenorth")
    {
        return std::shared_ptr<SomaModel>(new TrueNorthModel(group_id, id));
    }
    else
    {
        throw std::invalid_argument("Model not supported.");
    }
}
