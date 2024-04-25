def process_file(input_filename, output_filename):
  """
  This was used for data processing purposes to get out input data.
  Used Google Gemini to generate this function.
  """
  with open(input_filename, "r") as input_file, open(output_filename, "w") as output_file:
    for line in input_file:
      processed_line = line.replace("Cammeo", "0").replace("Osmancik", "1").strip()  # Replace and strip whitespace
      if processed_line:  # Check if line is not empty
        output_file.write(processed_line + "\n")

input = "Rice_Cammeo_Osmancik.arff"
output = "delicious_rice.txt"
process_file(input, output)
