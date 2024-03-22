from flask import Flask, jsonify, request
from flask_restful import Api, Resource

import components.gector.predict as gector


app = Flask(__name__)
api = Api(app)


model_gector_roberta = gector.load_for_demo()


class MODEL(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        model = json_data['model']
        text_input_list = json_data['text_input_list']
        print(f'======INPUT TO {model}=====', flush=True)
        print(text_input_list, sep='\n', flush=True)
        if model == 'GECToR-Roberta':
            text_output_list = gector.predict_for_demo(text_input_list, model_gector_roberta)
        elif model == 'GECToR-XLNet':
            text_output_list = "Unsupported"
        elif model == 'T5-Large':
            text_output_list = "Unsupported"
        else:
            raise NotImplementedError(f'Model {model} is not recognized.')        
        print(f'======OUTPUT FROM {model}=====', flush=True)
        print(text_output_list, sep='\n', flush=True)
        return jsonify({'model' : model, 'text_output_list' : text_output_list})


api.add_resource(MODEL, "/components/model")


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=3000, use_reloader=False)
