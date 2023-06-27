#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <torch/torch.h>

class Tokenizer {
public:
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    int vocab_size;

    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    }

    std::string detokenize(const std::vector<std::string>& tokens) {
        std::string text;
        for (const auto& token : tokens) {
            text += token + " ";
        }
        return text;
    }

    void build_vocab(const std::vector<std::string>& texts) {
        for (const auto& text : texts) {
            auto tokens = tokenize(text);
            for (const auto& token : tokens) {
                if (token_to_id.find(token) == token_to_id.end()) {
                    token_to_id[token] = vocab_size;
                    id_to_token[vocab_size] = token;
                    vocab_size++;
                }
            }
        }
    }

    int get_token_id(const std::string& token) {
        return token_to_id[token];
    }

    std::string get_token_from_id(int token_id) {
        return id_to_token[token_id];
    }
};

class GPTImpl : public torch::nn::Module {
public:
    GPTImpl(int vocab_size, int embedding_dim, int num_heads, int num_layers)
        : embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim)),
          pos_encoder(PositionalEncoding(embedding_dim)),
          transformer_encoder(torch::nn::TransformerEncoderOptions(embedding_dim, num_heads).num_layers(num_layers)),
          fc(embedding_dim, vocab_size) {
        register_module("embedding", embedding);
        register_module("pos_encoder", pos_encoder);
        register_module("transformer_encoder", transformer_encoder);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor input) {
        auto embedded = embedding(input);
        embedded = pos_encoder(embedded);
        auto encoded = transformer_encoder(embedded);
        auto output = fc(encoded);
        return output;
    }

private:
    torch::nn::Embedding embedding;
    PositionalEncoding pos_encoder;
    torch::nn::TransformerEncoder transformer_encoder;
    torch::nn::Linear fc;
};

class PositionalEncodingImpl : public torch::nn::Module {
public:
    PositionalEncodingImpl(int d_model, float dropout = 0.1, int max_len = 5000)
        : dropout(torch::nn::DropoutOptions(dropout)),
          positional_encoding(generate_positional_encoding(d_model, max_len)) {
        register_module("dropout", dropout);
    }

    torch::Tensor generate_positional_encoding(int d_model, int max_len) {
        torch::Tensor pe = torch::zeros({ max_len, d_model });
        auto position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
        auto div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) * (-std::log(10000.0) / d_model));
        pe.index({ torch::Slice(), torch::Slice(0, -1, 2) }) = torch::sin(position * div_term);
        pe.index({ torch::Slice(), torch::Slice(1, -1, 2) }) = torch::cos(position * div_term);
        pe = pe.unsqueeze(0).transpose(0, 1);
        return pe;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x + positional_encoding.index({ torch::Slice(0, x.size(0)), torch::Slice() });
        return dropout(x);
    }

private:
    torch::nn::Dropout dropout;
    torch::Tensor positional_encoding;
};

class GPT {
public:
    GPT(int vocab_size, int embedding_dim, int num_heads, int num_layers)
        : vocab_size(vocab_size),
          embedding_dim(embedding_dim),
          num_heads(num_heads),
          num_layers(num_layers),
          tokenizer(),
          model(nullptr),
          memory() {}

    std::vector<int> tokenize(const std::string& text) {
        auto tokens = tokenizer.tokenize(text);
        std::vector<int> token_ids;
        for (const auto& token : tokens) {
            token_ids.push_back(tokenizer.get_token_id(token));
        }
        return token_ids;
    }

    std::string generate_text(const std::string& user_input, int max_length = 100) {
        model->eval();
        torch::NoGradGuard no_grad;
        std::vector<int> input_ids = tokenize(user_input);
        torch::Tensor input_tensor = torch::tensor(input_ids).unsqueeze(1);
        std::vector<int> output_ids;
        for (int i = 0; i < max_length; ++i) {
            torch::Tensor output = model->forward(input_tensor);
            auto predicted_token = std::get<1>(output[-1].max(1));
            int predicted_token_value = predicted_token.item<int>();
            output_ids.push_back(predicted_token_value);
            input_ids.push_back(predicted_token_value);
            input_tensor = torch::tensor(input_ids).unsqueeze(1);
        }
        std::vector<std::string> output_tokens;
        for (const auto& token_id : output_ids) {
            output_tokens.push_back(tokenizer.get_token_from_id(token_id));
        }
        std::string generated_text = tokenizer.detokenize(output_tokens);
        return generated_text;
    }

    void save_tokenizer(const std::string& filepath) {
        torch::save(tokenizer, filepath);
    }

    void load_tokenizer(const std::string& filepath) {
        torch::load(tokenizer, filepath);
    }

    void build_vocab(const std::vector<std::string>& dataset) {
        tokenizer.build_vocab(dataset);
    }

    void initialize_model() {
        model = std::make_shared<GPTImpl>(vocab_size, embedding_dim, num_heads, num_layers);
    }

private:
    int vocab_size;
    int embedding_dim;
    int num_heads;
    int num_layers;
    Tokenizer tokenizer;
    std::shared_ptr<GPTImpl> model;
    std::shared_ptr<torch::Tensor> memory;
};

int main() {
    int vocab_size = 3285;
    int embedding_dim = 128;
    int num_heads = 4;
    int num_layers = 2;
    GPT gpt(vocab_size, embedding_dim, num_heads, num_layers);
    std::string tokenizer_filepath = "tokenizer.pth";
    std::string dataset_file = "dataset.txt";
    std::ifstream dataset_stream(dataset_file);
    std::vector<std::string> dataset;
    std::string line;
    while (std::getline(dataset_stream, line)) {
        dataset.push_back(line);
    }
    dataset_stream.close();

    gpt.build_vocab(dataset);
    gpt.initialize_model();
    gpt.save_tokenizer(tokenizer_filepath);

    while (true) {
        std::string user_input;
        std::cout << "Enter a prompt: ";
        std::getline(std::cin, user_input);
        if (user_input == "exit") {
            break;
        }
        std::string generated_text = gpt.generate_text(user_input);
        std::cout << "Generated text: " << generated_text << std::endl;
    }

    return 0;
}
