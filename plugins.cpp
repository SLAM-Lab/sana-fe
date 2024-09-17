// Copyright (c) 2024 - The University of Texas at Austin
//  This work was produced under contract #2317831 to National Technology and
//  Engineering Solutions of Sandia, LLC which is under contract
//  No. DE-NA0003525 with the U.S. Department of Energy.
//  plugins.cpp
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include <dlfcn.h>

#include "models.hpp"
#include "plugins.hpp"
#include "print.hpp"

using _create_synapse = sanafe::SynapseUnit *();
using _create_dendrite = sanafe::DendriteUnit *();
using _create_soma = sanafe::SomaUnit *();

std::map<std::string, _create_synapse *> plugin_create_synapse;
std::map<std::string, _create_dendrite *> plugin_create_dendrite;
std::map<std::string, _create_soma *> plugin_create_soma;

void sanafe::plugin_init_synapse(
        const std::string &model_name, const std::filesystem::path &plugin_path)
{
    const std::string create = "create_" + model_name;

    // Load the soma library
    INFO("Loading synapse plugin:%s\n", plugin_path.c_str());
    void *synapse = dlopen(plugin_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (synapse == nullptr)
    {
        INFO("Error: Couldn't load library %s\n", plugin_path.c_str());
        throw std::runtime_error("Error: Could not load library.\n");
    }

    // Reset DLL errors
    dlerror();

    // Function to create an instance of the Soma class
    INFO("Loading function: %s\n", create.c_str());
    auto *create_func = (_create_synapse *) dlsym(synapse, create.c_str());
    plugin_create_synapse[model_name] = create_func;

    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        INFO("Error: Couldn't load symbol %s: %s\n", create.c_str(),
                dlsym_error);
        throw std::runtime_error("Error: Could not load symbol.\n");
    }
    INFO("Loaded plugin symbols for %s.\n", model_name.c_str());
}

void sanafe::plugin_init_dendrite(
        const std::string &model_name, const std::filesystem::path &plugin_path)
{
    const std::string create = "create_" + model_name;

    // Load the soma library
    INFO("Loading dendrite plugin:%s\n", plugin_path.c_str());
    void *dendrite = dlopen(plugin_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (dendrite == nullptr)
    {
        INFO("Error: Couldn't load library %s\n", plugin_path.c_str());
        throw std::runtime_error("Error: Could not load library.\n");
    }

    // Reset DLL errors
    dlerror();

    // Function to create an instance of the Soma class
    INFO("Loading function: %s\n", create.c_str());
    auto *create_func = (_create_dendrite *) dlsym(dendrite, create.c_str());
    plugin_create_dendrite[model_name] = create_func;

    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        INFO("Error: Couldn't load symbol %s: %s\n", create.c_str(),
                dlsym_error);
        throw std::runtime_error("Error: Could not load symbol.\n");
    }

    INFO("Loaded plugin symbols for %s.\n", model_name.c_str());
}

void sanafe::plugin_init_soma(
        const std::string &model_name, const std::filesystem::path &plugin_path)
{
    const std::string create = "create_" + model_name;

    // Load the soma library
    INFO("Loading soma plugin:%s\n", plugin_path.c_str());
    void *soma = dlopen(plugin_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (soma == nullptr)
    {
        INFO("Error: Couldn't load library %s\n", plugin_path.c_str());
        throw std::invalid_argument("Error: Could not load library.\n");
    }

    // Reset DLL errors
    dlerror();

    // Function to create an instance of the Soma class
    INFO("Loading function: %s\n", create.c_str());
    auto *create_func = (_create_soma *) dlsym(soma, create.c_str());
    plugin_create_soma[model_name] = create_func;

    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        INFO("Error: Couldn't load symbol %s: %s\n", create.c_str(),
                dlsym_error);
        throw std::runtime_error("Error: Could not load symbol.\n");
    }

    INFO("Loaded plugin symbols for %s.\n", model_name.c_str());
}

std::shared_ptr<sanafe::SynapseUnit> sanafe::plugin_get_synapse(
        const std::string &model_name,
        const std::filesystem::path &plugin_path)
{
    if (plugin_path.empty())
    {
        throw std::runtime_error("No plugin path given.");
    }

    TRACE1("Getting synapse:%s\n", model_name.c_str());
    if (plugin_create_synapse.count(model_name) == 0)
    {
        plugin_init_synapse(model_name, plugin_path);
    }

    return std::shared_ptr<SynapseUnit>(
            plugin_create_synapse[model_name]());
}

std::shared_ptr<sanafe::DendriteUnit> sanafe::plugin_get_dendrite(
        const std::string &model_name, const std::filesystem::path &plugin_path)
{
    if (plugin_path.empty())
    {
        throw std::runtime_error("No plugin path given.");
    }

    TRACE1("Getting dendrite:%s\n", model_name.c_str());
    if (plugin_create_dendrite.count(model_name) == 0)
    {
        plugin_init_dendrite(model_name, plugin_path);
    }

    return std::shared_ptr<DendriteUnit>(plugin_create_dendrite[model_name]());
}

std::shared_ptr<sanafe::SomaUnit> sanafe::plugin_get_soma(
        const std::string &model_name,
        const std::filesystem::path &plugin_path)
{
    if (plugin_path.empty())
    {
        throw std::runtime_error("No plugin path given.");
    }

    TRACE1("Getting soma:%s\n", model_name.c_str());
    if (plugin_create_soma.count(model_name) == 0)
    {
        plugin_init_soma(model_name, plugin_path);
    }

    return std::shared_ptr<SomaUnit>(plugin_create_soma[model_name]());
}
