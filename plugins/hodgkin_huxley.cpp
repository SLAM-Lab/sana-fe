// Copyright (c) 2024 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
// hodgkin_huxley.cpp
// Plugin implementation of the Hodgkin-Huxley neuron model. Implemented by
//  Robin Sam.
// Model inspired by this paper: https://ieeexplore.ieee.org/document/9235538
//  and this textbook: https://mrgreene09.github.io/computational-neuroscience-textbook
#include <cmath>
#include <cstring>
#include <iostream>



#include "arch.hpp"
#include "models.hpp"
#include "plugins.hpp"
#include "print.hpp"

class HH : public sanafe::SomaUnit
{
    // HH specific
public:
    // system variables
    double C_m;
    double g_Na;
    double g_K;
    double g_L;
    double V_Na;
    double V_K;
    double V_L;
    double dt;

    // main parameters
    double V;
    double prev_V; // Membrane potential
    double I; // Stimulation current per area
    double m; // m, n, h are coeff
    double n;
    double h;

    // internal results of various differential equations
    double alpha_m;
    double alpha_n;
    double alpha_h;
    double beta_m;
    double beta_n;
    double beta_h;

    double tau_m;
    double tau_n;
    double tau_h;
    double pm;
    double pn;
    double ph;
    double denominator;
    double tau_V;
    double Vinf;

    HH()
    {
        V = 0.0;
        C_m = 10.0; // Effective capacitance per area of membrane; default is 1
        g_Na = 1200.0; // Conductance of sodium
        g_K = 360.0; // Conductance of potassium
        g_L = 3.0; // Conductance of leak channel
        V_Na = 50.0; // Reverse potential of sodium
        V_K = -77.0; // Reverse potential of potassium
        V_L = -54.387; // Reverse potential of leak channel
        dt = 0.1;
    }

    double get_potential(const size_t neuron_address) override
    {
        return V;
    }

    void reset() override
    {
        prev_V = 0.0;
        V = 0.0;
        m = 0.0;
        n = 0.0;
        h = 0.0;
        tau_n = 0.0;
        tau_m = 0.0;
        tau_h = 0.0;
        pm = 0.0;
        pn = 0.0;
        ph = 0.0;
        denominator = 0.0;
        tau_V = 0.0;
        Vinf = 0.0;

        return;
    }

    void set_attribute(const size_t neuron_address,
            const std::string &param_name,
            const sanafe::ModelParam &param) override
    {
        if (param_name == "m")
        {
            m = static_cast<double>(param);
        }
        else if (param_name == "n")
        {
            n = static_cast<double>(param);
        }
        else if (param_name == "h")
        {
            h = static_cast<double>(param);
        }
        else if (param_name == "current")
        {
            I = static_cast<double>(param);
        }
    }

    sanafe::SomaUnit::SomaResult update(const size_t neuron_address,
            const std::optional<double> current_in) override
    {
        sanafe::NeuronStatus status = sanafe::IDLE;

        // Calculate the change in potential since the last update e.g.
        //  integate inputs and apply any potential leak
        TRACE1(MODELS, "Updating potential, before:%f\n", V);

        alpha_n = (0.01 * (V + 55)) / (1 - exp(-0.1 * (V + 55)));
        alpha_m = (0.1 * (V + 40)) / (1 - exp(-0.1 * (V + 40)));
        alpha_h = 0.07 * exp(-0.05 * (V + 65));

        beta_n = 0.125 * exp(-0.01125 * (V + 55));
        beta_m = 4 * exp(-0.05556 * (V + 65));
        beta_h = 1 / (1 + exp(-0.1 * (V + 35)));

        tau_n = 1 / (alpha_n + beta_n);
        tau_m = 1 / (alpha_m + beta_m);
        tau_h = 1 / (alpha_h + beta_h);

        pm = alpha_m / (alpha_m + beta_m);
        pn = alpha_n / (alpha_n + beta_n);
        ph = alpha_h / (alpha_h + beta_h);

        denominator = g_L + g_K * (pow(n, 4)) + g_Na * (pow(m, 3) * h);
        tau_V = C_m / denominator;
        Vinf = ((g_L) *V_L + g_K * (pow(n, 4)) * V_K +
                       g_Na * (pow(m, 3)) * h * V_Na + I) /
                denominator;

        // update main parameters
        prev_V = V;
        V = Vinf + (V - Vinf) * exp(-1 * dt / tau_V);
        m = pm + (m - pm) * exp(-1 * dt / tau_m);
        n = pn + (n - pn) * exp(-1 * dt / tau_n);
        h = ph + (h - ph) * exp(-1 * dt / tau_h);

        // Check against threshold potential (for spiking)
        if ((prev_V < 25) && (V > 25))
        {
            // If voltage just crossed the 25 mV boundary, then
            //  spike
            status = sanafe::FIRED;
        }
        else
        {
            status = sanafe::UPDATED;
        }

        INFO("Updating potential, after:%f\n", V);

        return {status, std::nullopt, std::nullopt};
    }
};

// the Class factories
extern "C" sanafe::SomaUnit *create_HH()
{
    TRACE1(MODELS, "Creating HH soma instance\n");
    return (sanafe::SomaUnit *) new HH();
}
