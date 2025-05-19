here is the fine tuned model links

llama 3.2 1B
	https://huggingface.co/heyIamUmair/llama3-legal-lora-4epoch

	from peft import PeftModel
	from transformers import AutoModelForCausalLM

	base_model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit")
	model = PeftModel.from_pretrained(base_model, "heyIamUmair/llama3-legal-lora-4epoch")


classification Model: legal bert base uncased
	https://huggingface.co/heyIamUmair/legal-distilbert-uncased

	here is how to use it:
	classifier_path = "bert-base-uncased-classifier"  # Update with your path
	classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
	classification_model = AutoModelForSequenceClassification.from_pretrained(classifier_path).to("cuda")
