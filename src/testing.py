from audio_processor import create_batch_from_id


data_ids = ['f', 'e', 'd', 'c', '0', '9', '8', '7', '6', '5', '4', '3', '2', '1'] 

def testing(model, data_id = data_ids, first_num=None):
   
    melgrams, labels = create_batch_from_id(data_id, first_num)
    return model.evaluate(melgrams, labels)
