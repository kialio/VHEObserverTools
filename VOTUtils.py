#!/usr/bin/env python

"""Various common functions and utilities for the VOT modules"""

__author__ = 'Jeremy S. Perkins (FSSC)'
__version__ = '0.0.1'

import logging

class EANotFound: pass

def initLogger(name):

    """Sets up and returns a properly configured logging object."""

    quickLogger = logging.getLogger(name)
    quickLogger.setLevel(logging.DEBUG)
    #Prevents duuplicate log entries after reinitialization.                                                        
    if(not quickLogger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        quickLogger.addHandler(ch)

    return quickLogger

def checkParams(paramTemplate, params, logger):

    for par in paramTemplate:
        try:
            params[par]
        except KeyError:
            logger.critical("You must include the "+ par +" parameter for this model.")
            raise

