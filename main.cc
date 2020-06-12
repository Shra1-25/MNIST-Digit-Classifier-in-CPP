#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[]) {
    std::cout<<"Welcome to the digit classifier."<<endl;    
    
    std::string graph_definition = "tf_1_graph.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

    // Set GPU options
    //graph::SetDefaultDevice("/gpu:0", &graph_def);
    //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    //opts.config.mutable_gpu_options()->set_allow_growth(true);

    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));

    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));
    std::cout<<"Done1"<<endl;
    // Initialize our variables
    //TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));
    std::cout<<"Done2"<<endl;
    Tensor tmp(DT_FLOAT, TensorShape({28, 28}));
    
    auto _XTensor = tmp.matrix<float>();
    
    std::ifstream  data("input_data0.csv");
    std::string line;
    int i_idx=0;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        //std::vector<float> parsedRow;
        int j_idx=0;
        
        while(std::getline(lineStream,cell,','))
        {
            _XTensor(i_idx,j_idx)=std::stof(cell);
            //parsedRow.push_back(std::stof(cell));
            j_idx++;
        }
        //X_vec.push_back(parsedRow);
        i_idx++;
    }
    std::cout<<"Reading input data done."<<endl;
    
    //_XTensor.setRandom();
    Tensor x(DT_FLOAT, TensorShape({1, 28, 28, 1}));
    if(!x.CopyFrom(tmp, TensorShape({1, 28, 28, 1}))){
      std::cout<<"Reshape not successfull."<<endl;
    }
    //std::copy_n(X_vec.begin(), X_vec.size(), _XTensor.flat<float>().data());
    //_YTensor.setRandom();
    std::cout<<"Done3"<<endl;
    TF_CHECK_OK(session->Run({{"x", x}/*, {"y", y}*/}, {"dense_2_out"}, {}, &outputs)); // Get output
    std::cout<<"Done4"<<endl;
    int max_idx=0;
    float max_out = outputs[0].matrix<float>()(0,0);
    std::cout << "Output 0: " <<  idx_out << std::endl;
    for (int idx=1;idx<10;idx++){
        float idx_out = outputs[0].matrix<float>()(0,idx);
        std::cout << "Output "<<i<<": " <<  idx_out << std::endl;
        if (idx_out>max_out){
            max_out=idx_out;
            max_idx=idx;
        }
    }
    std::cout<<"The digit is: "<<max_idx<<endl;
    //TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr)); // Train
    outputs.clear();
    

    session->Close();
    delete session;
    //std::cout<<_YTensor(0,0)<<" "<<_YTensor(0,1)<<" "<<_YTensor(0,2)<<" "<<_YTensor(0,3)<<" "<<_YTensor(0,4)<<" "<<_YTensor(0,5)<<" "<<_YTensor(0,6)<<" "<<_YTensor(0,7)<<" "<<_YTensor(0,8)<<" "<<_YTensor(0,9)<<endl;
    //std::cout<<_YTensor(1,0)<<" "<<_YTensor(1,1)<<" "<<_YTensor(1,2)<<" "<<_YTensor(1,3)<<" "<<_YTensor(1,4)<<" "<<_YTensor(1,5)<<" "<<_YTensor(1,6)<<" "<<_YTensor(1,7)<<" "<<_YTensor(1,8)<<" "<<_YTensor(1,9)<<endl;
    std::cout<<"All done"<<endl;
    return 0;
}
