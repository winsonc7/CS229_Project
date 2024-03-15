import xml.etree.ElementTree as ET
import json
import sys

def main(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize a list to store the converted questions
    questions = []

    # Iterate over each Problem element
    for problem in root.findall('.//Problem'):
        # Extract relevant information
        subject = problem.find('Solution-Type').text
        body = problem.find('Body').text
        question = problem.find('Question').text
        grade = problem.attrib['Grade']
        # Concatenate body and question
        concatenated_question = f"{body} {question}"

        # Create a dictionary for the question
        question_dict = {
            "subject": subject,
            "question": concatenated_question,
            "grade": grade
        }

        # Append the question dictionary to the list
        questions.append(question_dict)

    # Write the questions to a JSON file
    json_file = f"{xml_path[:-4]}.json"
    with open(json_file, 'w') as f:
        json.dump(questions, f, indent=4)

# Usage: python xml_to_json.py ASDiv.xml
if __name__ == "__main__":
    xml_path = sys.argv[1]
    main(xml_path)