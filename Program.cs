using Microsoft.ML.OnnxRuntimeGenAI;
using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;
using System.Text;

namespace MyNewCopilot_PC;

internal class Program
{
	// Download from https://huggingface.co/microsoft/
	// Phi-3.5-mini-instruct-onnx/tree/main
	private static readonly string modelDir =
	  @"C:\Code\Phi-3.5-mini-instruct-onnx\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4";

	static async Task Main(string[] args)
	{
		Console.WriteLine($"Loading model: {modelDir}");
		var sw = Stopwatch.StartNew();
		using var model = new Model(modelDir);
		using var tokenizer = new Tokenizer(model);
		sw.Stop();
		Console.WriteLine($"Model loading took {sw.ElapsedMilliseconds} ms");

		await ListIcao("KDFW", model, tokenizer);
		await ListIcao("KCLE", model, tokenizer);
	}

	private static async Task<bool> ListIcao(string icaoName, Model model, Tokenizer tokenizer)
	{
		var systemPrompt = "You are a helpful assistant.";
		var icao = Icao.Instance.GetIcao(icaoName);
		var userPrompt = $"Tell me about {icao.City} {icao.State} . Be brief.";

		var prompt = $"<|system|>{systemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>";
		await WriteResults(prompt, model, tokenizer);

		Console.WriteLine();
		Console.WriteLine("");

		userPrompt = $"Tell me about ICAO {icao.ICAO}.  Be concise. ";

		prompt = $"<|system|>{systemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>";
		await WriteResults(prompt, model, tokenizer);

		Console.WriteLine();
		return true;
	}
	private static async Task<bool> WriteResults(string prompt, Model model, Tokenizer tokenizer)
	{
		await foreach (var part in InferStreaming(prompt, model, tokenizer))
		{
			Console.Write(part);
		}

		return true;
	}

	public static async IAsyncEnumerable<string> InferStreaming(string prompt,
	  Model model, Tokenizer tokenizer)
	{
		using var generatorParams = new GeneratorParams(model);
		using var sequences = tokenizer.Encode(prompt);
		generatorParams.SetSearchOption("max_length", 2048);
		generatorParams.SetSearchOption("top_p", 0.5);
		//generatorParams.SetSearchOption("top_k", 1);
		generatorParams.SetSearchOption("temperature", 0.8);
		generatorParams.SetInputSequences(sequences);
		generatorParams.TryGraphCaptureWithMaxBatchSize(1);

		using var tokenizerStream = tokenizer.CreateStream();
		using var generator = new Generator(model, generatorParams);
		StringBuilder stringBuilder = new();

		while (!generator.IsDone())
		{
			string part;
			await Task.Delay(10).ConfigureAwait(false);
			generator.ComputeLogits();
			generator.GenerateNextToken();
			part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
			stringBuilder.Append(part);

			if (stringBuilder.ToString().Contains("<|end|>")
			  || stringBuilder.ToString().Contains("<|user|>")
			  || stringBuilder.ToString().Contains("<|system|>"))
			{
				break;
			}

			if (!string.IsNullOrWhiteSpace(part))
				yield return part;
		}
	}
}

