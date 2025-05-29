#include "YoloDetector.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace sycl;
using namespace yolo;
using std::vector;
typedef vector<float> FloatVector;
typedef vector<int16_t> Int16Vector;

void YoloDetector::loadWeights(const std::string& weightFile) {
    std::ifstream file(weightFile, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open weights file: " + weightFile);
    }

    file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));

    // instead of dividing input by 255, we divide the weight of first layer by 255.
    for (int j = weightOffsets[0]; j < biasOffsets[0]; j++) {
        weights[j] /= 255.0f;
    }
    // Calculate max weights and quantize
    float prodOfMaxWOfLayer = 1.0f;
    for (int i = 0; i < 9; i++) {
        float maxWeight = 0.0f;
        for (int j = weightOffsets[i]; j < biasOffsets[i]; j++) {
            maxWeight = std::max(maxWeight, std::abs(weights[j]));
        }

        prodOfMaxWOfLayer *= maxWeight;
        maxWeight /= 256.0f;
        maxWeightDivBy256[i] = maxWeight;

        quantizedWeights_low[i].resize(biasOffsets[i] - weightOffsets[i]);
        quantizedBiases_low[i].resize(layerOutputChannels[i]);
        quantizedBiases_high[i].resize(layerOutputChannels[i]);

        for (int j = weightOffsets[i]; j < biasOffsets[i]; j++) {
            int32_t w = static_cast<int32_t>(std::round(weights[j] / maxWeight));
            //Replacing 32 bit int with 2 16 bit int
            //quantizedWeights_high[j] = 0; //the weights are designed to be between [-256,256]
            quantizedWeights_low[i][j - weightOffsets[i]] = w & 0xFFFF;
        }

        float scaleForBias = prodOfMaxWOfLayer / (1 << exponentsForBias[i]);

        for (int j = biasOffsets[i]; j < weightOffsets[i + 1]; j++) {
            int32_t w = static_cast<int32_t>(std::round(weights[j] / scaleForBias));
            //Replacing 32 bit int with 2 16 bit int
            quantizedBiases_high[i][j - biasOffsets[i]] = w >> 16;
            quantizedBiases_low[i][j - biasOffsets[i]] = w & 0xFFFF;
        }
    }
}

void YoloDetector::runConvolutionLayers(queue& q) {
    int16_t sumOfShifts = 0;
    const int16_t SubThreadPart = SubThreadPart;
    try {
        // Create two SYCL buffers for input & output feature maps, 
        // so we can do the "swap" trick after each layer.
        buffer<int16_t, 2> bufferInput(inputFeatures.data(), range<2>(inputFeatures.size() / SubThreadPart, SubThreadPart));
        // Weights as one large buffer:
        buffer<int16_t, 2> bufferConv_low(convolutionSums_low.data(), range<2>(convolutionSums_low.size() / SubThreadPart, SubThreadPart));
        buffer<int16_t, 2> bufferConv_high(convolutionSums_high.data(), range<2>(convolutionSums_high.size() / SubThreadPart, SubThreadPart));

        buffer<int16_t, 2> bufferActivation14_1(activation14_1.data(), range<2>(activation14_1.size() / SubThreadPart, SubThreadPart));
        buffer<int16_t, 2> bufferActivation14_2(activation14_2.data(), range<2>(activation14_2.size() / SubThreadPart, SubThreadPart));

        // Loop over all layers
        for (int16_t layer = 0; layer < 9; layer++) {

            lengthOfActLimit = lengthsOfActLimit[layer];//D
            exponentForBias = exponentsForBias[layer];//I
            shift4Bias4Sum = exponentForBias - 8 * (layer + 1) + sumOfShifts;//M
            shift4Bias14 = (layer == 0) ? 2 : (17 - shift4Bias4Sum);//J
            realLenOfAct = (layer == 0) ? 8 : ((length14 + shift4Sum14 > lengthOfActLimit) ? lengthOfActLimit : (length14 + shift4Sum14 > lengthOfActLimit));//E
            shift4Conv14 = (layer == 0) ? 7 : (17 - lengthOfActLimit + realLenOfAct);//F
            shift4Sum14 = std::max(shift4Conv14, shift4Bias14);//K
            int16_t shift4Bias14Sum = shift4Sum14 + shift4Bias4Sum;

            int16_t szIn = layerInputDims[layer];
            int16_t szOut = layerOutputDims[layer];
            int16_t channelsIn = layerInputChannels[layer];
            int16_t channelsOut = layerOutputChannels[layer];
            int16_t pool = poolKernelSizes[layer];
            int16_t conv = convKernelSizes[layer];
            //size_t weightOffset = weightOffsets[layer];
            //size_t biasOffset = biasOffsets[layer];
            int16_t totalThreads = (((channelsOut * szOut) / 7) * szOut) / 7;

            //buffer<int16_t, 1> subbufferWeights_high(bufferWeights_high, id<1>(weightOffset), range<1>(biasOffset - weightOffset));
            buffer<int16_t, 2> bufferWeights_low(quantizedWeights_low[layer].data(), range<2>(channelsOut, conv * conv * channelsIn));
            buffer<int16_t, 1> bufferBiases_high(quantizedBiases_high[layer].data(), range<1>(channelsOut));
            buffer<int16_t, 1> bufferBiases_low(quantizedBiases_low[layer].data(), range<1>(channelsOut));

            // Submit a command group for the parallel-for
            q.submit([&](handler& cgh) {
                // Accessors for read/write
                auto accessorInput = bufferInput.get_access<access::mode::read>(cgh);
                //auto accessorWeights_high = bufferWeights_high2d.get_access<access::mode::read>(cgh);
                auto accessorWeights_low = bufferWeights_low.get_access<access::mode::read>(cgh);
                auto accessorBiases_high = bufferBiases_high.get_access<access::mode::read>(cgh);
                auto accessorBiases_low = bufferBiases_low.get_access<access::mode::read>(cgh);
                // No-init or discard_write on the output if you don’t need old data
                auto accessorOutput_low = bufferConv_low.get_access<access::mode::write>(cgh);
                auto accessorOutput_high = bufferConv_high.get_access<access::mode::write>(cgh);
                auto accessorOutput14 = bufferActivation14_1.get_access<access::mode::write>(cgh);

                // We launch 1D range = (#output_channels * layerOutputDims[layer] * layerOutputDims[layer])
                // so each work-item corresponds to one (channel_output, x_output, y_output).

                cgh.parallel_for(range<2>(totalThreads, SubThreadPart), [=](id<2> threadIdx) {
                    // Decompose threadIdx into (channel_output, x_output, y_output)
                    int16_t channelOutput, xOutput = 0, yOutput = 0;
                    if (layer < 8) {
                        channelOutput = (threadIdx[0] / (szOut / 7)) / (szOut / 7);
                        xOutput = (threadIdx[0] / szOut) * SubThreadPart + ((threadIdx[0] % szOut) * SubThreadPart + threadIdx[1]) / szOut;
                        xOutput -= channelOutput * szOut;
                        yOutput = ((threadIdx[0] % szOut) * SubThreadPart + threadIdx[1]) % szOut;
                    }
                    else {
                        channelOutput = threadIdx[0] * SubThreadPart + threadIdx[1];
                    }

                    // We'll track the maximum over the pooling region
                    int16_t poolingMaxValue_high = -32768;
                    int16_t poolingMaxValue_low = 0x0000;

                    // Max-pooling: loop over xPooling and yPooling
                    for (int16_t xPooling = 0; xPooling < pool; xPooling++) {
                        for (int16_t yPooling = 0; yPooling < pool; yPooling++) {

                            int16_t convolutionSum_high = 0, convolutionSum_low = 0;

                            // Convolution kernel loop
                            for (int16_t xKernel = 0; xKernel < conv; xKernel++) {
                                for (int16_t yKernel = 0; yKernel < conv; yKernel++) {

                                    // Map these into input coordinates
                                    int16_t xInput = static_cast<int16_t>(
                                        pool * xOutput + xPooling + xKernel
                                        - conv / 2
                                        );
                                    int16_t yInput = static_cast<int16_t>(
                                        pool * yOutput + yPooling + yKernel
                                        - conv / 2
                                        );

                                    // Check bounds (in-bounds vs. out-of-bounds)
                                    bool inBounds = (xInput >= 0 && xInput < szIn &&
                                        yInput >= 0 && yInput < szIn);

                                    // For each input channel
                                    for (int16_t channelInput = 0; channelInput < channelsIn; channelInput++) {
                                        // Compute input index (assuming layout is:
                                        //   channel*(width*height) + x*(height) + y)

                                        int16_t xClamped = (xInput < 0) ? 0 : xInput;
                                        xClamped = (xClamped > szIn - 1) ? (szIn - 1) : xClamped;
                                        int16_t yClamped = (yInput < 0) ? 0 : yInput;
                                        yClamped = (yClamped > szIn - 1) ? (szIn - 1) : yClamped;

                                        int16_t inputIndex0 =
                                            (channelInput * szIn + xClamped) / SubThreadPart * szIn
                                            + (((channelInput * szIn + xClamped) % SubThreadPart) * szIn + yClamped) / SubThreadPart;
                                        int16_t inputIndex1 =
                                            +(((channelInput * szIn + xClamped) % SubThreadPart) * szIn + yClamped) % SubThreadPart;

                                        int16_t inputVal = inBounds ? accessorInput[inputIndex0][inputIndex1] : 0;

                                        int16_t weightIndex = channelInput * conv * conv
                                            + xKernel * conv
                                            + yKernel;

                                        // -------------------------------------------------
                                        // Integer multiplication 16(activation) * 8(weight)
                                        // -------------------------------------------------
                                        int16_t w = accessorWeights_low[channelOutput][weightIndex];
                                        int16_t a = inputVal;

                                        int16_t A_low8 = static_cast<int16_t>(a & 0xFF);           // Extract the low 16 bits of firstInt
                                        int16_t A_high8 = static_cast<int16_t>((a >> 8) & 0xFF);  // Extract the high 16 bits of firstInt

                                        int16_t product_low8 = w * A_low8;
                                        int16_t product_high8 = w * A_high8;
                                        int16_t a_low, b_low, a_high, b_high;


                                        // Check for overflow or carry in the low part and handle accordingly:
                                        // 1. If both low parts are negative, there might be a carry to the high part.
                                        // 2. If one low part is negative and the other is positive, we check for overflow or carry based on the magnitude of the numbers.
                                        // ----    End    -----------------------------

                                        // --------------------------------------------
                                        // 32bit addition    conv += w * a
                                        // --------------------------------------------
                                        // ----    End    -----------------------------
                                    } // channelInput
                                } // yKernel
                            } // xKernel

                            // Update max for pooling
                            // --------------------------------------------
                            // 32bit comparison    maxpool = max(pool, conv)
                            // --------------------------------------------

                            // ----    End    -----------------------------

                        } // yPooling
                    } // xPooling

                    // Finally, store the maximum pooled value
                    accessorOutput_high[threadIdx] = poolingMaxValue_high;
                    accessorOutput_low[threadIdx] = poolingMaxValue_low;

                    // Calculate Sum14
                    // --------------------------------------------
                    // right shift operation for 
                    // sum14 = (int16_t)(poolingMaxValue >> shift4Sum14);
                    // --------------------------------------------
                    int16_t sum14;
                    if (shift4Sum14 > 15)
                        sum14 = poolingMaxValue_high >> (shift4Sum14 - 16);
                    else {
                        sum14 = ((1 << (16 - shift4Sum14)) - 1) & (poolingMaxValue_low >> shift4Sum14) | (poolingMaxValue_high << (16 - shift4Sum14));
                    }

                    if (shift4Bias14Sum < 32) {
                        // --------------------------------------------
                        // right shift operation for 
                        // sum14 += (int16_t)(accessorWeights[biasOffset + channelOutput] >> shift4Bias14Sum);
                        // --------------------------------------------
                        int16_t bias_high = accessorBiases_high[channelOutput];
                        int16_t bias_low = accessorBiases_low[channelOutput];
                        if (shift4Bias14Sum > 15)
                            sum14 += bias_high >> (shift4Bias14Sum - 16);
                        else {
                            sum14 += ((1 << (16 - shift4Bias14Sum)) - 1) & (bias_low >> shift4Bias14Sum) | (bias_high << (16 - shift4Bias14Sum));
                        }
                    }

                    // Leaky ReLU (alpha=0.1)
                    if (sum14 < 0) {
                        sum14 /= -10;
                    }
                    accessorOutput14[threadIdx] = sum14;

                    }); // end parallel_for
                }); // end submit

            // Wait for the layer to finish before we swap
            q.wait_and_throw();

            // -------------GET MAXIMUM
            // Pointers to buffers, for swapping after each layer
            buffer<int16_t, 2>* maxBuffer1 = &bufferActivation14_1;
            buffer<int16_t, 2>* maxBuffer2 = &bufferActivation14_2;
            int16_t N = totalThreads / 2;
            while (N > 0) {
                q.submit([&](sycl::handler& h) {
                    auto accessor1 = maxBuffer1->get_access<access::mode::read>(h);
                    auto accessor2 = maxBuffer2->get_access<access::mode::write>(h);
                    h.parallel_for(range<2>(N, SubThreadPart), [=](id<2> idx) {
                        accessor2[idx] = std::max(accessor1[2 * idx[0]][idx[1]], accessor1[2 * idx[0] + 1][idx[1]]);
                        }); // end parallel_for
                    }); // end submit
                q.wait_and_throw();
                if (N == 1)
                    break;
                // Swap the buffers so the output of this layer is the input to the next
                std::swap(maxBuffer1, maxBuffer2);
                N = (N + 1) / 2;  // Half the number of elements after each stage
            }
            N = SubThreadPart / 2;
            while (N > 0) {
                q.submit([&](sycl::handler& h) {
                    auto accessor1 = maxBuffer1->get_access<access::mode::read>(h);
                    auto accessor2 = maxBuffer2->get_access<access::mode::write>(h);
                    h.parallel_for(range<2>(1, N), [=](id<2> idx) {
                        accessor2[idx] = std::max(accessor1[idx[0]][idx[1] * 2], accessor1[idx[0]][idx[1] * 2 + 1]);
                        }); // end parallel_for
                    }); // end submit
                q.wait_and_throw();
                if (N == 1)
                    break;
                // Swap the buffers so the output of this layer is the input to the next
                std::swap(maxBuffer1, maxBuffer2);
                N = (N + 1) / 2;  // Half the number of elements after each stage
            }
            // getting the maximum

            //getting the binary length of max14
            length14 = std::log2(max14) + 1;

            //std::cout << "length14: " << length14 << std::endl;
            // -------------Making Activation with proper range
            shift4Layer = std::max(length14 + shift4Sum14 - lengthsOfActLimit[layer + 1], 0);
            //std::cout << "L shift for layer is: " << shift4Layer << std::endl;
            sumOfShifts += shift4Layer;
            //std::cout << "shift4Bias4Sum: " << shift4Bias4Sum << std::endl;
            int16_t shift4Bias = shift4Layer + shift4Bias4Sum;
            //std::cout << "shift for bias for sum is: " << shift4Bias << std::endl;

            // Wait for the layer to finish before we swap
            q.wait_and_throw();

        } // end for (layer)
    }
    catch (std::exception const& e) {
        std::cerr << "An exception was caught in runConvolutionLayers(): "
            << e.what() << "\n";
        std::terminate();
    }
}
