"""
convert.py

Author: Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script contains the Convert_YCB class which is used for converting between different naming conventions
in the YCB dataset. The class provides methods for converting between the original object names and
their corresponding descriptions, as well as between object names and their numerical identifiers.

The class also provides methods for retrieving the list of all object names and their corresponding descriptions
in the dataset.

The conversion mappings are defined in a static method and stored in dictionaries for efficient lookup.
The class also includes some basic tests to verify the correctness of the conversions.
"""

class Convert_YCB:
    def __init__(self):
        self.conversion_dict, self.name_to_number_dict, self.number_to_name_dict = self.create_conversion_dict()
        self.object_list = list(self.name_to_number_dict.keys())
        self.desc_names_list = [self.convert_name(name) for name in self.object_list]

    def convert_name(self, input_string):
        # Return the converted string if available, otherwise return the original string
        return self.conversion_dict.get(input_string)

    def convert_number(self, input_value):
        # Determine if the input is a name or a number and convert accordingly
        if isinstance(input_value, str):
            return self.name_to_number_dict.get(input_value)
        elif isinstance(input_value, int):
            return self.number_to_name_dict.get(input_value)
        else:
            raise ValueError("The input should either be a string or an int")

    def get_object_list(self):
        return self.object_list

    def get_desc_names_list(self):
        return self.desc_names_list

    @staticmethod
    def create_conversion_dict():
        # Original mapping provided in the question
        mapping_text = """
        002_master_chef_can blue cylindrical can
        003_cracker_box red cracker cardbox
        004_sugar_box yellow sugar cardbox
        005_tomato_soup_can red cylindrical can
        006_mustard_bottle yellow mustard bottle
        007_tuna_fish_can tuna fish tin can
        008_pudding_box brown jelly cardbox
        009_gelatin_box red jelly cardbox
        010_potted_meat_can spam rectangular can
        011_banana banana
        019_pitcher_base blue cup
        021_bleach_cleanser white bleach bottle
        024_bowl red bowl
        025_mug red cup
        035_power_drill drill
        036_wood_block wooden block
        037_scissors scissors
        040_large_marker marker pen
        051_large_clamp black clamp
        052_extra_large_clamp bigger black clamp
        061_foam_brick red rectangular block
        """
        # Splitting the mapping text into lines and then into key-value pairs
        pairs = mapping_text.strip().split('\n')
        conversion_dict = {}
        name_to_number = {}
        number_to_name = {}

        for idx, pair in enumerate(pairs, start=1):
            parts = pair.split(maxsplit=1)  # Split the line at the first space
            if len(parts) == 2:
                key = parts[0]  # First part
                value = parts[1]  # Second part
                conversion_dict[key] = value
                conversion_dict[value] = key  # This line allows bidirectional lookup
                name_to_number[key] = idx
                number_to_name[idx] = key

        return conversion_dict, name_to_number, number_to_name


if __name__ == '__main__':
    convert_string = Convert_YCB()
    assert convert_string.convert_name("002_master_chef_can") == "blue_cylindrical_can"
    assert convert_string.convert_name("blue_cylindrical_can") == "002_master_chef_can"
    assert convert_string.convert_name("banana") == "011_banana"
    assert convert_string.convert_name("011") is None
