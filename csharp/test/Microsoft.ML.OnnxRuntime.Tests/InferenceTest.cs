// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    public class InferenceTest
    {
        private const string module = "onnxruntime.dll";
        private const string propertiesFile = "Properties.txt";
        private readonly ITestOutputHelper output;

        public InferenceTest(ITestOutputHelper o)
        {
            this.output = o;
        }

        [Fact]
        public void TestSessionOptions()
        {
            using (SessionOptions opt = new SessionOptions())
            {
                Assert.NotNull(opt);

                // check default values of the properties
                Assert.Equal(ExecutionMode.ORT_SEQUENTIAL, opt.ExecutionMode);
                Assert.True(opt.EnableMemoryPattern);
                Assert.False(opt.EnableProfiling);
                Assert.Equal("onnxruntime_profile_", opt.ProfileOutputPathPrefix);
                Assert.True(opt.EnableCpuMemArena);
                Assert.Equal("", opt.LogId);
                Assert.Equal(LogLevel.Verbose, opt.LogVerbosityLevel);
                Assert.Equal(0, opt.IntraOpNumThreads);
                Assert.Equal(0, opt.InterOpNumThreads);
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_BASIC, opt.GraphOptimizationLevel);

                // try setting options 
                opt.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                Assert.Equal(ExecutionMode.ORT_PARALLEL, opt.ExecutionMode);

                opt.EnableMemoryPattern = false;
                Assert.False(opt.EnableMemoryPattern);

                opt.EnableProfiling = true;
                Assert.True(opt.EnableProfiling);
                Assert.Equal("onnxruntime_profile_", opt.ProfileOutputPathPrefix);

                opt.ProfileOutputPathPrefix = "Ort_P_";
                Assert.Equal("Ort_P_", opt.ProfileOutputPathPrefix);

                opt.EnableCpuMemArena = false;
                Assert.False(opt.EnableCpuMemArena);

                opt.LogId = "MyLogId";
                Assert.Equal("MyLogId", opt.LogId);

                opt.LogVerbosityLevel = LogLevel.Error;
                Assert.Equal(LogLevel.Error, opt.LogVerbosityLevel);

                opt.IntraOpNumThreads = 4;
                Assert.Equal(4, opt.IntraOpNumThreads);

                opt.InterOpNumThreads = 4;
                Assert.Equal(4, opt.InterOpNumThreads);

                opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                Assert.Equal(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, opt.GraphOptimizationLevel);

                Assert.Throws<OnnxRuntimeException>(() => { opt.GraphOptimizationLevel = (GraphOptimizationLevel)10; });

                opt.AppendExecutionProvider_CPU(1);
#if USE_MKLDNN
                opt.AppendExecutionProvider_Mkldnn(0);
#endif
#if USE_CUDA
                opt.AppendExecutionProvider_CUDA(0);
#endif
#if USE_NGRAPH
                opt.AppendExecutionProvider_NGraph("CPU");  //TODO: this API should be refined
#endif
#if USE_OPENVINO
                opt.AppendExecutionProvider_OpenVINO(null);  //TODO: this won't work, because the native side copies the const char*
#endif
#if USE_TENSORRT
                opt.AppendExecutionProvider_Tensorrt(0);
#endif
#if USE_NNAPI
                opt.AppendExecutionProvider_Nnapi();
#endif


            }
        }

        [Fact]
        public void TestRunOptions()
        {
            using (var opt = new RunOptions())
            {
                Assert.NotNull(opt);

                //verify default options
                Assert.False(opt.Terminate);
                Assert.Equal(LogLevel.Verbose, opt.LogVerbosityLevel);
                Assert.Equal("", opt.LogId);

                // try setting options
                opt.Terminate = true;
                Assert.True(opt.Terminate);

                opt.LogVerbosityLevel = LogLevel.Error;
                Assert.Equal(LogLevel.Error, opt.LogVerbosityLevel);

                opt.LogId = "MyLogTag";
                Assert.Equal("MyLogTag", opt.LogId);
            }
        }

        [Fact]
        public void CanCreateAndDisposeSessionWithModelPath()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                Assert.NotNull(session);
                Assert.NotNull(session.InputMetadata);
                Assert.Equal(1, session.InputMetadata.Count); // 1 input node
                Assert.True(session.InputMetadata.ContainsKey("data_0")); // input node name
                Assert.Equal(typeof(float), session.InputMetadata["data_0"].ElementType);
                Assert.True(session.InputMetadata["data_0"].IsTensor);
                var expectedInputDimensions = new int[] { 1, 3, 224, 224 };
                Assert.Equal(expectedInputDimensions.Length, session.InputMetadata["data_0"].Dimensions.Length);
                for (int i = 0; i < expectedInputDimensions.Length; i++)
                {
                    Assert.Equal(expectedInputDimensions[i], session.InputMetadata["data_0"].Dimensions[i]);
                }

                Assert.NotNull(session.OutputMetadata);
                Assert.Equal(1, session.OutputMetadata.Count); // 1 output node
                Assert.True(session.OutputMetadata.ContainsKey("softmaxout_1")); // output node name
                Assert.Equal(typeof(float), session.OutputMetadata["softmaxout_1"].ElementType);
                Assert.True(session.OutputMetadata["softmaxout_1"].IsTensor);
                var expectedOutputDimensions = new int[] { 1, 1000, 1, 1 };
                Assert.Equal(expectedOutputDimensions.Length, session.OutputMetadata["softmaxout_1"].Dimensions.Length);
                for (int i = 0; i < expectedOutputDimensions.Length; i++)
                {
                    Assert.Equal(expectedOutputDimensions[i], session.OutputMetadata["softmaxout_1"].Dimensions[i]);
                }
            }
        }

        [Theory]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, true)]
        [InlineData(GraphOptimizationLevel.ORT_DISABLE_ALL, false)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, true)]
        [InlineData(GraphOptimizationLevel.ORT_ENABLE_EXTENDED, false)]
        private void CanRunInferenceOnAModel(GraphOptimizationLevel graphOptimizationLevel, bool enableParallelExecution)
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");

            // Set the graph optimization level for this session.
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = graphOptimizationLevel;
            if (enableParallelExecution) options.ExecutionMode = ExecutionMode.ORT_PARALLEL;

            using (var session = new InferenceSession(modelPath, options))
            {
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();

                float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model

                foreach (var name in inputMeta.Keys)
                {
                    Assert.Equal(typeof(float), inputMeta[name].ElementType);
                    Assert.True(inputMeta[name].IsTensor);
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run the inference
                using (var results = session.Run(container))  // results is an IReadOnlyList<NamedOnnxValue> container
                {
                    validateRunResults(results);
                }

                // Run Inference with RunOptions
                using (var runOptions = new RunOptions())
                {
                    runOptions.LogId = "CsharpTest";
                    runOptions.Terminate = false;  // TODO: Test terminate = true, it currently crashes
                    runOptions.LogVerbosityLevel = LogLevel.Error;
                    IReadOnlyCollection<string> outputNames = session.OutputMetadata.Keys.ToList();

                    using (var results = session.Run(container, outputNames, runOptions))  // results is an IReadOnlyList<NamedOnnxValue> container
                    {
                        validateRunResults(results);
                    }
                }
            }
        }

        private void validateRunResults(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
            // validate the results
            foreach (var r in results)
            {
                Assert.Equal(1, results.Count);
                Assert.Equal("softmaxout_1", r.Name);

                var resultTensor = r.AsTensor<float>();
                int[] expectedDimensions = { 1, 1000, 1, 1 };  // hardcoded for now for the test data
                Assert.Equal(expectedDimensions.Length, resultTensor.Rank);

                var resultDimensions = resultTensor.Dimensions;
                for (int i = 0; i < expectedDimensions.Length; i++)
                {
                    Assert.Equal(expectedDimensions[i], resultDimensions[i]);
                }

                var resultArray = r.AsTensor<float>().ToArray();
                Assert.Equal(expectedOutput.Length, resultArray.Length);
                Assert.Equal(expectedOutput, resultArray, new floatComparer());
            }
        }


        [Fact]
        private void ThrowWrongInputName()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("wrong_name", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.Contains("Invalid Feed Input", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void ThrowWrongInputType()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            int[] inputDataInt = inputData.Select(x => (int)x).ToArray();
            var tensor = new DenseTensor<int>(inputDataInt, inputMeta["data_0"].Dimensions);
            container.Add(NamedOnnxValue.CreateFromTensor<int>("data_0", tensor));
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            var msg = ex.ToString().Substring(0, 101);
            // TODO: message is diff in LInux. Use substring match
            Assert.Equal("Microsoft.ML.OnnxRuntime.OnnxRuntimeException: [ErrorCode:InvalidArgument] Unexpected input data type", msg);
            session.Dispose();
        }

        [Fact]
        private void ThrowExtraInputs()
        {
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            var nov1 = NamedOnnxValue.CreateFromTensor<float>("data_0", tensor);
            var nov2 = NamedOnnxValue.CreateFromTensor<float>("extra", tensor);
            container.Add(nov1);
            container.Add(nov2);
            var ex = Assert.Throws<OnnxRuntimeException>(() => session.Run(container));
            Assert.StartsWith("[ErrorCode:InvalidArgument] Invalid Feed Input Name", ex.Message);
            session.Dispose();
        }

        [Fact]
        private void TestMultiThreads()
        {
            var numThreads = 10;
            var loop = 10;
            var tuple = OpenSessionSqueezeNet();
            var session = tuple.Item1;
            var inputData = tuple.Item2;
            var tensor = tuple.Item3;
            var expectedOut = tuple.Item4;
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
            var tasks = new Task[numThreads];
            for (int i = 0; i < numThreads; i++)
            {
                tasks[i] = Task.Factory.StartNew(() =>
                {
                    for (int j = 0; j < loop; j++)
                    {
                        var resnov = session.Run(container);
                        var res = resnov.ToArray()[0].AsTensor<float>().ToArray<float>();
                        Assert.Equal(res, expectedOut, new floatComparer());
                    }
                });
            };
            Task.WaitAll(tasks);
            session.Dispose();
        }

        private static Dictionary<string, string> GetSkippedModels()
        {
            var skipModels = new Dictionary<string, string>() {
                { "mxnet_arcface", "Model is an invalid ONNX model"},
                { "tf_inception_v2", "TODO: Debug failing model, skipping for now" },
                { "fp16_inception_v1", "16-bit float not supported type in C#." },
                { "fp16_shufflenet", "16-bit float not supported type in C#." },
                { "fp16_tiny_yolov2", "16-bit float not supported type in C#." },
                { "BERT_Squad", "Could not find an implementation for the node bert / embeddings / one_hot:OneHot(9)" },
                { "mlperf_ssd_mobilenet_300", "Could not find file output_0.pb" },
                { "tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied" },
                { "tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied" }
            };

            // The following models fails on nocontribops win CI
            var disableContribOpsEnvVar = Environment.GetEnvironmentVariable("DisableContribOps");
            var isContribOpsDisabled = (disableContribOpsEnvVar != null) ? disableContribOpsEnvVar.Equals("ON") : false;
            if (isContribOpsDisabled)
            {
                skipModels["test_tiny_yolov2"] = "Fails when ContribOps is disabled";
                skipModels["mask_rcnn_keras"] = "Pad is not a registered function/op";
            }

            // This model fails on x86 Win CI
            if (System.Environment.Is64BitProcess == false)
            {
                skipModels["test_vgg19"] = "Get preallocated buffer for initializer conv4_4_b_0 failed";
                skipModels["tf_pnasnet_large"] = "Get preallocated buffer for initializer ConvBnFusion_BN_B_cell_5/comb_iter_1/left/bn_sep_7x7_1/beta:0_203 failed";
                skipModels["tf_nasnet_large"] = "Get preallocated buffer for initializer ConvBnFusion_BN_B_cell_11/beginning_bn/beta:0_331 failed";
            }

            return skipModels;
        }

        public static IEnumerable<object[]> GetModelsForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels();

            foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
            {
                //var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opsetDir.Name));
                foreach (var modelDir in opsetDir.EnumerateDirectories())
                {
                    if (!skipModels.ContainsKey(modelDir.Name))
                    {
                        yield return new object[] { modelDir.Parent.Name, modelDir.Name };
                    }
                } //model
            } //opset
        }

        public static IEnumerable<object[]> GetSkippedModelForTest()
        {
            var modelsDir = GetTestModelsDir();
            var modelsDirInfo = new DirectoryInfo(modelsDir);
            var skipModels = GetSkippedModels();

            foreach (var opsetDir in modelsDirInfo.EnumerateDirectories())
            {
                var modelRoot = new DirectoryInfo(Path.Combine(modelsDir, opsetDir.Name));
                foreach (var modelDir in modelRoot.EnumerateDirectories())
                {
                    if (skipModels.ContainsKey(modelDir.Name))
                    {
                        //Console.WriteLine("Model {0} is skipped due to the error: {1}", modelDir.FullName, skipModels[modelDir.Name]);
                        yield return new object[] { modelDir.Parent.Name, modelDir.Name };
                    }

                }
            }
        }


        [Theory]
        [MemberData(nameof(GetModelsForTest))]
        [MemberData(nameof(GetSkippedModelForTest), Skip = "Skipped due to Error, please fix the error and enable the test")]
        private void TestPreTrainedModels(string opset, string modelName)
        {
            var modelsDir = GetTestModelsDir();
            string onnxModelFileName = null;

            var modelDir = new DirectoryInfo(Path.Combine(modelsDir, opset, modelName));

            try
            {
                var onnxModelNames = modelDir.GetFiles("*.onnx");
                bool validModelFound = false;
                if (onnxModelNames.Length > 0)
                {
                    // TODO remove file "._resnet34v2.onnx" from test set
                    for (int i = 0; i < onnxModelNames.Length; i++)
                    {
                        if (onnxModelNames[i].Name != "._resnet34v2.onnx")
                        {
                            onnxModelNames[0] = onnxModelNames[i];
                            validModelFound = true;
                        }
                    }
                }

                if (validModelFound)
                {
                    onnxModelFileName = Path.Combine(modelDir.FullName, onnxModelNames[0].Name);
                }
                else
                {
                    var modelNamesList = string.Join(",", onnxModelNames.Select(x => x.ToString()));
                    throw new Exception($"Opset {opset} Model {modelName}. Can't determine model file name. Found these :{modelNamesList}");
                }

                using (var session = new InferenceSession(onnxModelFileName))
                {
                    var inMeta = session.InputMetadata;
                    string testDataDirNamePattern = "test_data*";
                    if (opset == "opset9" && modelName == "LSTM_Seq_lens_unpacked")
                    {
                        testDataDirNamePattern = "seq_lens*"; // discrepency in data directory
                    }
                    var testDataDir = modelDir.EnumerateDirectories(testDataDirNamePattern).First();
                    var inputContainer = new List<NamedOnnxValue>();
                    var outputContainer = new List<NamedOnnxValue>();
                    foreach (var f in testDataDir.EnumerateFiles("input_*.pb"))
                    {
                        inputContainer.Add(LoadTensorFromFilePb(f.FullName, inMeta));
                    }
                    foreach (var f in testDataDir.EnumerateFiles("output_*.pb"))
                    {
                        outputContainer.Add(LoadTensorFromFilePb(f.FullName, session.OutputMetadata));
                    }

                    using (var resultCollection = session.Run(inputContainer))
                    {
                        foreach (var result in resultCollection)
                        {
                            Assert.True(session.OutputMetadata.ContainsKey(result.Name));
                            var outputMeta = session.OutputMetadata[result.Name];
                            NamedOnnxValue outputValue = null;
                            foreach (var o in outputContainer)
                            {
                                if (o.Name == result.Name)
                                {
                                    outputValue = o;
                                    break;
                                }
                            }
                            if (outputValue == null)
                            {
                                outputValue = outputContainer.First(); // in case the output data file does not contain the name
                            }
                            if (outputMeta.IsTensor)
                            {
                                if (outputMeta.ElementType == typeof(float))
                                {
                                    Assert.Equal(result.AsTensor<float>(), outputValue.AsTensor<float>(), new floatComparer());
                                }
                                else if (outputMeta.ElementType == typeof(int))
                                {
                                    Assert.Equal(result.AsTensor<int>(), outputValue.AsTensor<int>(), new ExactComparer<int>());
                                }
                                else if (outputMeta.ElementType == typeof(uint))
                                {
                                    Assert.Equal(result.AsTensor<uint>(), outputValue.AsTensor<uint>(), new ExactComparer<uint>());
                                }
                                else if (outputMeta.ElementType == typeof(short))
                                {
                                    Assert.Equal(result.AsTensor<short>(), outputValue.AsTensor<short>(), new ExactComparer<short>());
                                }
                                else if (outputMeta.ElementType == typeof(ushort))
                                {
                                    Assert.Equal(result.AsTensor<ushort>(), outputValue.AsTensor<ushort>(), new ExactComparer<ushort>());
                                }
                                else if (outputMeta.ElementType == typeof(long))
                                {
                                    Assert.Equal(result.AsTensor<long>(), outputValue.AsTensor<long>(), new ExactComparer<long>());
                                }
                                else if (outputMeta.ElementType == typeof(ulong))
                                {
                                    Assert.Equal(result.AsTensor<ulong>(), outputValue.AsTensor<ulong>(), new ExactComparer<ulong>());
                                }
                                else if (outputMeta.ElementType == typeof(byte))
                                {
                                    Assert.Equal(result.AsTensor<byte>(), outputValue.AsTensor<byte>(), new ExactComparer<byte>());
                                }
                                else if (outputMeta.ElementType == typeof(bool))
                                {
                                    Assert.Equal(result.AsTensor<bool>(), outputValue.AsTensor<bool>(), new ExactComparer<bool>());
                                }
                                else
                                {
                                    Assert.True(false, "The TestPretrainedModels does not yet support output of type " + nameof(outputMeta.ElementType));
                                }
                            }
                            else
                            {
                                Assert.True(false, "TestPretrainedModel cannot handle non-tensor outputs yet");
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                var msg = $"Opset {opset}, Model {modelName}: ModelFile = {onnxModelFileName} error = {ex.Message}";
                throw new Exception(msg + "\n" + ex.StackTrace);
            }
        }

        [Fact]
        private void TestOverridableInitializerMetadata()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "overridable_initializer.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                Assert.Equal(2, session.InputMetadata.Count);
                Assert.True(session.InputMetadata.ContainsKey("Label"));
                Assert.True(session.InputMetadata.ContainsKey("F2"));

                Assert.Equal(1, session.OverridableInitializerMetadata.Count);
                Assert.True(session.OverridableInitializerMetadata.ContainsKey("F1"));
                Assert.True(session.OverridableInitializerMetadata["F1"].IsTensor);
                Assert.Equal(typeof(float), session.OverridableInitializerMetadata["F1"].ElementType);
                Assert.Equal(2, session.OverridableInitializerMetadata["F1"].Dimensions.Length);
                Assert.Equal(1, session.OverridableInitializerMetadata["F1"].Dimensions[0]);
                Assert.Equal(1, session.OverridableInitializerMetadata["F1"].Dimensions[1]);

                var container = new List<NamedOnnxValue>();
                var Label_input = new DenseTensor<bool>(new bool[] { true }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("Label", Label_input));

                var F2_input = new DenseTensor<string>(new string[] { "f2_string" }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("F2", F2_input));

                var F1_initializer = new DenseTensor<float>(new float[] { 2.0f }, new int[] { 1, 1 });
                container.Add(NamedOnnxValue.CreateFromTensor("F1", F1_initializer));

                using (var result = session.Run(container))
                {
                    var resultMap = new Dictionary<string, NamedOnnxValue>();

                    foreach (var output in result)
                    {
                        resultMap[output.Name] = output;
                    }

                    Assert.True(resultMap.ContainsKey("Label0"));
                    Assert.True(resultMap.ContainsKey("F20"));
                    Assert.True(resultMap.ContainsKey("F11"));

                    var overriddenInitializer = resultMap["F11"].AsTensor<float>();
                    Assert.NotNull(overriddenInitializer);
                    Assert.True(overriddenInitializer.SequenceEqual(F1_initializer));
                }
            }
        }

        [Fact]
        private void TestSymbolicDimsMetadata()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "capi_symbolic_dims.onnx");
            using (var session = new InferenceSession(modelPath))
            {
                var inputs = session.InputMetadata;
                var outputs = session.OutputMetadata;

                Assert.Equal(2, inputs.Count);
                Assert.Equal(1, session.OutputMetadata.Count);
                Assert.True(inputs.ContainsKey("A"));
                Assert.True(inputs.ContainsKey("B"));
                Assert.True(outputs.ContainsKey("C"));

                var inputA = inputs["A"];
                var inputB = inputs["B"];
                var outputC = outputs["C"];

                // dimension values and any symbolic dimension info should have the same length
                Assert.Equal(inputA.Dimensions.Length, inputA.SymbolicDimensions.Length);
                Assert.Equal(inputB.Dimensions.Length, inputB.SymbolicDimensions.Length);
                Assert.Equal(outputC.Dimensions.Length, outputC.SymbolicDimensions.Length);

                Assert.Equal(inputA.Dimensions, new int[] { -1, 2 });
                Assert.Equal(inputA.SymbolicDimensions, new string[] { "n", "" });
                Assert.Equal(inputB.Dimensions, new int[] { -1 });
                Assert.Equal(inputB.SymbolicDimensions, new string[] { "m" });
                Assert.Equal(outputC.Dimensions, new int[] { -1 });
                Assert.Equal(outputC.SymbolicDimensions, new string[] { "" }); // unnamed symbolic dim
            }
        }


        [Fact]
        private void TestModelInputFloat()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT.pb");

            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<float>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputBOOL()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_BOOL.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<bool>(new bool[] { true, false, true, false, true }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<bool>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT32.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<int>(new int[] { 1, -2, -3, int.MinValue, int.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<int>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputDOUBLE()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_DOUBLE.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<double>(new double[] { 1.0, 2.0, -3.0, 5, 5 }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<double>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }

        }

        [Fact]
        private void TestModelInputSTRING()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_STRING.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<string>(new string[] { "abc", "ced", "def", "", "frozen" }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<string>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT8.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<sbyte>(new sbyte[] { 1, 2, -3, sbyte.MinValue, sbyte.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<sbyte>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT8()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT8.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<byte>(new byte[] { 1, 2, 3, byte.MinValue, byte.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<byte>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt16>(new UInt16[] { 1, 2, 3, UInt16.MinValue, UInt16.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Int16>(new Int16[] { 1, 2, 3, Int16.MinValue, Int16.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<Int16>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_INT64.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<Int64>(new Int64[] { 1, 2, -3, Int64.MinValue, Int64.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<Int64>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelInputUINT32()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT32.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt32>(new UInt32[] { 1, 2, 3, UInt32.MinValue, UInt32.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt32>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }
        [Fact]
        private void TestModelInputUINT64()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_UINT64.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<UInt64>(new UInt64[] { 1, 2, 3, UInt64.MinValue, UInt64.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<UInt64>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact(Skip = "FLOAT16 not available in C#")]
        private void TestModelInputFLOAT16()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT16.pb");
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<float>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [Fact]
        private void TestModelSequenceOfMapIntFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_sequence_map_int_float.pb");
            using (var session = new InferenceSession(modelPath))
            {

                var outMeta = session.OutputMetadata;
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outMeta["label"].OnnxValueType);
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outMeta["probabilities"].OnnxValueType);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode1 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode1.Name);

                    // try-cast as a tensor
                    var outLabelTensor = outNode1.AsTensor<Int64>();

                    // Label 1 should have highest probaility
                    Assert.Equal(1, outLabelTensor[0]);

                    // second output is a sequence<map<int64, float>>
                    // try-cast to an sequence of NOV
                    var outNode2 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode2.Name);

                    // try-cast to an sequence of NOV
                    var seq = outNode2.AsEnumerable<NamedOnnxValue>();

                    // try-cast first element in sequence to map/dictionary type
                    if (System.Environment.Is64BitProcess)
                    {
                        var map = seq.First().AsDictionary<Int64, float>();
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                    else // 32-bit
                    {
                        var map = seq.First().AsDictionary<long, float>();
                        Assert.Equal(0.25938290, map[0], 6);
                        Assert.Equal(0.40904793, map[1], 6);
                        Assert.Equal(0.33156919, map[2], 6);
                    }
                }
            }
        }

        [Fact]
        private void TestModelSequenceOfMapStringFloat()
        {
            // test model trained using lightgbm classifier
            // produces 2 named outputs
            //   "label" is a tensor,
            //   "probabilities" is a sequence<map<int64, float>>
            // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_sequence_map_string_float.pb");

            using (var session = new InferenceSession(modelPath))
            {
                var outMeta = session.OutputMetadata;
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outMeta["label"].OnnxValueType);
                Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, outMeta["probabilities"].OnnxValueType);

                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 5.8f, 2.8f }, new int[] { 1, 2 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);

                using (var outputs = session.Run(container))
                {
                    // first output is a tensor containing label
                    var outNode1 = outputs.ElementAtOrDefault(0);
                    Assert.Equal("label", outNode1.Name);

                    // try-cast as a tensor
                    var outLabelTensor = outNode1.AsTensor<string>();

                    // Label 1 should have highest probaility
                    Assert.Equal("1", outLabelTensor[0]);

                    // second output is a sequence<map<int64, float>>
                    // try-cast to an sequence of NOV
                    var outNode2 = outputs.ElementAtOrDefault(1);
                    Assert.Equal("probabilities", outNode2.Name);

                    // try-cast to an sequence of NOV
                    var seq = outNode2.AsEnumerable<NamedOnnxValue>();

                    // try-cast first element in sequence to map/dictionary type
                    var map = seq.First().AsDictionary<string, float>();
                    //verify values are valid
                    Assert.Equal(0.25938290, map["0"], 6);
                    Assert.Equal(0.40904793, map["1"], 6);
                    Assert.Equal(0.33156919, map["2"], 6);
                }
            }
        }

        [Fact(Skip = "The Model Serialization Test fails on linux. Test skipped until fixed. Serialization API should not be used before fix.")]
        private void TestModelSerialization()
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            string modelOutputPath = Path.Combine(Directory.GetCurrentDirectory(), "optimized-squeezenet.onnx");
            // Set the optimized model file path to assert that no exception are thrown.
            SessionOptions options = new SessionOptions();
            options.OptimizedModelFilePath = modelOutputPath;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
            var session = new InferenceSession(modelPath, options);
            Assert.NotNull(session);
            Assert.True(File.Exists(modelOutputPath));
        }

        [GpuFact]
        private void TestGpu()
        {
            var gpu = Environment.GetEnvironmentVariable("TESTONGPU");
            var tuple = OpenSessionSqueezeNet(0); // run on deviceID 0
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");

            using (var session = tuple.Item1)
            {
                var inputData = tuple.Item2;
                var tensor = tuple.Item3;
                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();
                container.Add(NamedOnnxValue.CreateFromTensor<float>("data_0", tensor));
                var res = session.Run(container);
                var resultArray = res.First().AsTensor<float>().ToArray();
                Assert.Equal(expectedOutput, resultArray, new floatComparer());
            }
        }

        [Fact]
        private void TestInferenceSessionWithByteArray()
        {
            // model takes 1x5 input of fixed type, echoes back
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "test_types_FLOAT.pb");
            byte[] modelData = File.ReadAllBytes(modelPath);

            using (var session = new InferenceSession(modelData))
            {
                var container = new List<NamedOnnxValue>();
                var tensorIn = new DenseTensor<float>(new float[] { 1.0f, 2.0f, -3.0f, float.MinValue, float.MaxValue }, new int[] { 1, 5 });
                var nov = NamedOnnxValue.CreateFromTensor("input", tensorIn);
                container.Add(nov);
                using (var res = session.Run(container))
                {
                    var tensorOut = res.First().AsTensor<float>();
                    Assert.True(tensorOut.SequenceEqual(tensorIn));
                }
            }
        }

        [DllImport("kernel32", SetLastError = true)]
        static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32", CharSet = CharSet.Ansi)]
        static extern UIntPtr GetProcAddress(IntPtr hModule, string procName);

        [Fact]
        private void VerifyNativeMethodsExist()
        {
            // Check for  external API changes
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return;
            var entryPointNames = new[]{
            "OrtGetApiBase",
            "OrtSessionOptionsAppendExecutionProvider_CPU"
#if USE_MKLDNN
            ,"OrtSessionOptionsAppendExecutionProvider_Mkldnn"
#endif
#if USE_CUDA
            ,"OrtSessionOptionsAppendExecutionProvider_CUDA"
#endif
#if USE_NGRAPH
            ,"OrtSessionOptionsAppendExecutionProvider_NGraph"
#endif
#if USE_OPENVINO
            ,"OrtSessionOptionsAppendExecutionProvider_OpenVINO"
#endif
#if USE_TENSORRT
            ,"OrtSessionOptionsAppendExecutionProvider_Tensorrt"
#endif
#if USE_NNAPI
            ,"OrtSessionOptionsAppendExecutionProvider_Nnapi"
#endif
    };

            var hModule = LoadLibrary(module);
            foreach (var ep in entryPointNames)
            {
                var x = GetProcAddress(hModule, ep);
                Assert.False(x == UIntPtr.Zero, $"Entrypoint {ep} not found in module {module}");
            }
        }

        static string GetTestModelsDir()
        {
            // get build directory, append downloaded models location
            var cwd = Directory.GetCurrentDirectory();
            var props = File.ReadAllLines(Path.Combine(cwd, propertiesFile));
            var modelsRelDir = Path.Combine(props[0].Split('=')[1].Trim());
            var modelsDir = Path.Combine(cwd, @"../../..", modelsRelDir, "models");
            return modelsDir;
        }

        static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }


        private enum TensorElementType
        {
            Float = 1,
            UInt8 = 2,
            Int8 = 3,
            UInt16 = 4,
            Int16 = 5,
            Int32 = 6,
            Int64 = 7,
            String = 8,
            Bool = 9,
            Float16 = 10,
            Double = 11,
            UInt32 = 12,
            UInt64 = 13,
            Complex64 = 14,
            Complex128 = 15,
            BFloat16 = 16,
            DataTypeMax = 17
        }

        private static void GetTypeAndWidth(TensorElementType elemType, out Type type, out int width)
        {
            switch (elemType)
            {
                case TensorElementType.Float:
                    type = typeof(float);
                    width = sizeof(float);
                    break;
                case TensorElementType.Double:
                    type = typeof(double);
                    width = sizeof(double);
                    break;
                case TensorElementType.Int16:
                    type = typeof(short);
                    width = sizeof(short);
                    break;
                case TensorElementType.UInt16:
                    type = typeof(ushort);
                    width = sizeof(ushort);
                    break;
                case TensorElementType.Int32:
                    type = typeof(int);
                    width = sizeof(int);
                    break;
                case TensorElementType.UInt32:
                    type = typeof(uint);
                    width = sizeof(uint);
                    break;
                case TensorElementType.Int64:
                    type = typeof(long);
                    width = sizeof(long);
                    break;
                case TensorElementType.UInt64:
                    type = typeof(ulong);
                    width = sizeof(ulong);
                    break;
                case TensorElementType.UInt8:
                    type = typeof(byte);
                    width = sizeof(byte);
                    break;
                case TensorElementType.Int8:
                    type = typeof(sbyte);
                    width = sizeof(sbyte);
                    break;
                case TensorElementType.String:
                    type = typeof(byte);
                    width = sizeof(byte);
                    break;
                case TensorElementType.Bool:
                    type = typeof(bool);
                    width = sizeof(bool);
                    break;
                default:
                    type = null;
                    width = 0;
                    break;
            }
        }
        static NamedOnnxValue LoadTensorFromFilePb(string filename, IReadOnlyDictionary<string, NodeMetadata> nodeMetaDict)
        {
            var file = File.OpenRead(filename);
            var tensor = Onnx.TensorProto.Parser.ParseFrom(file);
            file.Close();

            Type tensorElemType = null;
            int width = 0;
            GetTypeAndWidth((TensorElementType)tensor.DataType, out tensorElemType, out width);
            var intDims = new int[tensor.Dims.Count];
            for (int i = 0; i < tensor.Dims.Count; i++)
            {
                intDims[i] = (int)tensor.Dims[i];
            }

            NodeMetadata nodeMeta = null;
            string nodeName = "";
            if (nodeMetaDict.Count == 1)
            {
                nodeMeta = nodeMetaDict.Values.First();
                nodeName = nodeMetaDict.Keys.First(); // valid for single node input
            }
            else if (nodeMetaDict.Count > 1)
            {
                if (tensor.Name != "")
                {
                    nodeMeta = nodeMetaDict[tensor.Name];
                    nodeName = tensor.Name;
                }
                else
                {
                    bool matchfound = false;
                    // try to find from matching type and shape
                    foreach (var key in nodeMetaDict.Keys)
                    {
                        var meta = nodeMetaDict[key];
                        if (tensorElemType == meta.ElementType && tensor.Dims.Count == meta.Dimensions.Length)
                        {
                            int i = 0;
                            for (; i < meta.Dimensions.Length; i++)
                            {
                                if (meta.Dimensions[i] != -1 && meta.Dimensions[i] != intDims[i])
                                {
                                    break;
                                }
                            }
                            if (i >= meta.Dimensions.Length)
                            {
                                matchfound = true;
                                nodeMeta = meta;
                                nodeName = key;
                                break;
                            }
                        }
                    }
                    if (!matchfound)
                    {
                        // throw error
                        throw new Exception("No Matching Tensor found in InputOutputMetadata corresponding to the serliazed tensor loaded from " + filename);
                    }
                }
            }
            else
            {
                // throw error
                throw new Exception("While reading the serliazed tensor loaded from " + filename + ", metaDataDict has 0 elements");
            }

            Assert.True(nodeMeta.IsTensor, "LoadTensorFromFile can load Tensor types only");
            //TODO: support other types when models are available in Onnx model zoo/ test data

            Assert.Equal(tensorElemType, nodeMeta.ElementType);
            Assert.Equal(nodeMeta.Dimensions.Length, tensor.Dims.Count);
            for (int i = 0; i < nodeMeta.Dimensions.Length; i++)
            {
                Assert.True((nodeMeta.Dimensions[i] == -1) || (nodeMeta.Dimensions[i] == intDims[i]));
            }

            if (nodeMeta.ElementType == typeof(float))
            {
                return CreateNamedOnnxValueFromRawData<float>(nodeName, tensor.RawData.ToArray(), sizeof(float), intDims);
            }
            else if (nodeMeta.ElementType == typeof(double))
            {
                return CreateNamedOnnxValueFromRawData<double>(nodeName, tensor.RawData.ToArray(), sizeof(double), intDims);
            }
            else if (nodeMeta.ElementType == typeof(int))
            {
                return CreateNamedOnnxValueFromRawData<int>(nodeName, tensor.RawData.ToArray(), sizeof(int), intDims);
            }
            else if (nodeMeta.ElementType == typeof(uint))
            {
                return CreateNamedOnnxValueFromRawData<uint>(nodeName, tensor.RawData.ToArray(), sizeof(uint), intDims);
            }
            else if (nodeMeta.ElementType == typeof(long))
            {
                return CreateNamedOnnxValueFromRawData<long>(nodeName, tensor.RawData.ToArray(), sizeof(long), intDims);
            }
            else if (nodeMeta.ElementType == typeof(ulong))
            {
                return CreateNamedOnnxValueFromRawData<ulong>(nodeName, tensor.RawData.ToArray(), sizeof(ulong), intDims);
            }
            else if (nodeMeta.ElementType == typeof(short))
            {
                return CreateNamedOnnxValueFromRawData<short>(nodeName, tensor.RawData.ToArray(), sizeof(short), intDims);
            }
            else if (nodeMeta.ElementType == typeof(ushort))
            {
                return CreateNamedOnnxValueFromRawData<ushort>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
            }
            else if (nodeMeta.ElementType == typeof(byte))
            {
                return CreateNamedOnnxValueFromRawData<byte>(nodeName, tensor.RawData.ToArray(), sizeof(byte), intDims);
            }
            else if (nodeMeta.ElementType == typeof(bool))
            {
                return CreateNamedOnnxValueFromRawData<bool>(nodeName, tensor.RawData.ToArray(), sizeof(bool), intDims);
            }
            else
            {
                //TODO: Add support for remaining types
                Assert.True(false, "Tensors of type " + nameof(nodeMeta.ElementType) + " not currently supported in the LoadTensorFromFile");
                return null;
            }
        }

        static NamedOnnxValue CreateNamedOnnxValueFromRawData<T>(string name, byte[] rawData, int elemWidth, int[] dimensions)
        {
            T[] floatArr = new T[rawData.Length / elemWidth];
            Buffer.BlockCopy(rawData, 0, floatArr, 0, rawData.Length);
            var dt = new DenseTensor<T>(floatArr, dimensions);
            return NamedOnnxValue.CreateFromTensor<T>(name, dt);
        }

        static Tuple<InferenceSession, float[], DenseTensor<float>, float[]> OpenSessionSqueezeNet(int? cudaDeviceId = null)
        {
            string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "squeezenet.onnx");
            var option = new SessionOptions();
#if USE_CUDA
            if (cudaDeviceId.HasValue)
            {
                option = SessionOptions.MakeSessionOptionWithCudaProvider(cudaDeviceId.Value);
            }
#endif
            var session = (cudaDeviceId.HasValue)
                ? new InferenceSession(modelPath, option)
                : new InferenceSession(modelPath);
            float[] inputData = LoadTensorFromFile(@"bench.in");
            float[] expectedOutput = LoadTensorFromFile(@"bench.expected_out");
            var inputMeta = session.InputMetadata;
            var tensor = new DenseTensor<float>(inputData, inputMeta["data_0"].Dimensions);
            return new Tuple<InferenceSession, float[], DenseTensor<float>, float[]>(session, inputData, tensor, expectedOutput);
        }

        class floatComparer : IEqualityComparer<float>
        {
            private float atol = 1e-3f;
            private float rtol = 1.7e-2f;

            public bool Equals(float x, float y)
            {
                return Math.Abs(x - y) <= (atol + rtol * Math.Abs(y));
            }
            public int GetHashCode(float x)
            {
                return 0;
            }
        }

        class ExactComparer<T> : IEqualityComparer<T>
        {
            public bool Equals(T x, T y)
            {
                return x.Equals(y);
            }
            public int GetHashCode(T x)
            {
                return 0;
            }
        }


        private class GpuFact : FactAttribute
        {
            public GpuFact()
            {
                var testOnGpu = System.Environment.GetEnvironmentVariable("TESTONGPU");
                if (testOnGpu == null || !testOnGpu.Equals("ON"))
                {
                    Skip = "GPU testing not enabled";
                }
            }
        }

    }
}
