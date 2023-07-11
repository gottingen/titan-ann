// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tann/utility/ArgumentsParser.h"

using namespace tann::Helper;


ArgumentsParser::IArgument::IArgument()
{
}


ArgumentsParser::IArgument::~IArgument()
{
}


ArgumentsParser::ArgumentsParser()
{
}


ArgumentsParser::~ArgumentsParser()
{
}


bool
ArgumentsParser::Parse(int p_argc, char** p_args)
{
    while (p_argc > 0)
    {
        int last = p_argc;
        for (auto& option : m_arguments)
        {
            if (!option->ParseValue(p_argc, p_args))
            {
                TLOG_TRACE("Failed to parse args around \"{}\"\n", *p_args);
                PrintHelp();
                return false;
            }
        }

        if (last == p_argc)
        {
            p_argc -= 1;
            p_args += 1;
        }
    }

    bool isValid = true;
    for (auto& option : m_arguments)
    {
        if (option->IsRequiredButNotSet())
        {
            TLOG_TRACE("Required option not set:\n  ");
            option->PrintDescription();
            TLOG_TRACE("\n");
            isValid = false;
        }
    }

    if (!isValid)
    {
        TLOG_TRACE("\n");
        PrintHelp();
        return false;
    }

    return true;
}


void
ArgumentsParser::PrintHelp()
{
    TLOG_TRACE("Usage: ");
    for (auto& option : m_arguments)
    {
        TLOG_TRACE("\n  ");
        option->PrintDescription();
    }

    TLOG_TRACE("\n\n");
}
