# modelop.schema.0: input_schema.avsc
# modelop.slot.1: in-use


import pickle
import pandas


# modelop.init
def begin()-> None:
    """A function to load globally the trained model.
    """

    global iris_classifier
    iris_classifier = pickle.load(open("iris_tree_classifier.pkl", "rb"))

# modelop.score
def action(data: dict)->dict:
    """A function to predict iris species.

    Args:
        data (dict): Input record respecting the input schema.

    Returns:
        dict: Output appended to input dict.
    """

    # Turn input dict into a 1-record DF
    data = pandas.DataFrame([data])

    # Add prediction column
    data['predicted_species'] = iris_classifier.predict(
        data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
    )

    # yield a dictionary
    return data.to_dict(orient="records")[0]

# TEST
if __name__=='__main__':
    data = pandas.read_json("baseline_data.json", lines=True, orient="records")
    begin()
    for _, rec in data.iterrows():
        print(action(rec))