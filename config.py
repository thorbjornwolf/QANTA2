"""This module wraps a local config file.
It is motivated by the need for host-specific paths to the Stanford Parser,
but may see other uses as well. A usage example:

>>> from config import get_config
>>> config = get_config('My section')
>>> stored_value = config['stored_value_key']
Config file: Section 'My section' does not have key 'stored_value_key'
Please enter the wanted value of key 'stored_value_key'
> 12345
Key 'stored_value_key' set to value '12345' in config file. Proceeding.
>>> stored_value
'12345'
"""


import ConfigParser

CONFIG_FILE = 'local_config'

class ConfigSection(object):
    
    def __init__(self, section):
        config = ConfigParser.ConfigParser()
        config.optionxform = str
        config.read(CONFIG_FILE)
        
        if not config.has_section(section):
            config.add_section(section)

        self.section = section
        self.config = config

    def __getitem__(self, key):
        if not self.config.has_option(self.section, key):
            print ("Config file: Section '{}' does not have key '{}'").format(self.section, key)

            question = "Please enter the wanted value of key '{}'\n> ".format(key)
            value = raw_input(question)
            self.config.set(self.section, key, value)
            write(self.config)
            
            print "Key '{}' set to value '{}' in config file. Proceeding.".format(key, value)
        return self.config.get(self.section, key)

def get_config(section):
    """Returns a dict of configuration keys, values
    If section is not defined in the config file, it takes
    the user through creating that section.

    Section can be a module name, or whatever makes sense when you create it.
    """
    return ConfigSection(section)

def write(config):
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)


def ask_and_insert_item(config, section, key):
    """Ask the user for a value to put in config's 'section' 
    under 'key':

    Paraphrased mental model:
    config[
        section[
            key:value (answer to 'question')
        ]
    ]
    """
    
