// Verifies the process-owned compatibility key shared by Fit and Post.
#include "core/Model.h"
#include "core/Model.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
{
    try {
        const ctpwa::ModelDefinition nominal =
            ctpwa::load_model_definition("config/model.json");
        const std::string implementation =
            gvv_model_implementation_signature(nominal);
        const std::string definition_signature =
            ctpwa::model_definition_signature(nominal);
        const std::string combined = gvv_model_signature(nominal);

        require(!implementation.empty(),
                "GVV implementation signature is empty");
        require(combined == implementation + ':' + definition_signature,
                "combined GVV model signature has the wrong contract");

        // FitState stores the canonical model as a structured JSON object.
        // Re-dumping that object must preserve the parsed model contract.
        const nlohmann::json embedded_document =
            nlohmann::json::parse(nominal.canonical_json);
        const ctpwa::ModelDefinition restored =
            ctpwa::parse_model_definition(
                embedded_document.dump(2) + '\n',
                "fit-state structured model round trip");
        require(ctpwa::model_definition_signature(restored)
                    == definition_signature,
                "structured model round trip changed its signature");

        nlohmann::json modified_document =
            nlohmann::json::parse(nominal.canonical_json);
        modified_document["metadata"]["description"] =
            "signature regression variant";
        const ctpwa::ModelDefinition modified =
            ctpwa::parse_model_definition(
                modified_document.dump(), "signature regression variant");
        require(gvv_model_signature(modified) != combined,
                "GVV model signature ignored a model-definition change");

        std::cout << "GVV model signature test passed\n";
    } catch (const std::exception& error) {
        std::cerr << "GVV model signature test failed: "
                  << error.what() << '\n';
        return 1;
    }
    return 0;
}
