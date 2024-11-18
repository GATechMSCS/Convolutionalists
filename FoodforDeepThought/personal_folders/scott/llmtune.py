from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LLMTune:
    def __init__():
        pass

    def bert():
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

        # Fine-tune the model on food-related QA dataset (not shown here)

        def get_food_info(food_item, question):
            inputs = tokenizer.encode_plus(question, food_item, return_tensors='pt')
            outputs = model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
            return answer

        # Example usage
        food_item = "apple"
        nutrition_facts = get_food_info(food_item, "What are the nutrition facts of an apple?")
        health_benefits = get_food_info(food_item, "What are the health benefits of eating apples?")
        recipe = get_food_info(food_item, "Give me a simple recipe using apples.")

    def gpt():
        

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

        def generate_food_info(prompt, max_length=150):
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, temperature=0.7)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Example usage
        food_item = "sushi"
        nutrition_info = generate_food_info(f"Nutrition facts of {food_item}:")
        health_benefits = generate_food_info(f"Health benefits of eating {food_item}:")
        recipe = generate_food_info(f"A simple recipe for {food_item}:")


def main():
    llm = LLMTune()

    llm.bert()

    llm.gpt()

if __name__ == "__main__":
    main()