#include <iostream>

// TODO:FLEXNLP change to array of FlexNLPAccelerators instead of LSTMAcc + Change include
#include "LSTMAcc.h"
// Class for keeping track of 1 flexnlp's context (operators declared)
class FlexNLPContext {
  private:
    // Accelerators' identifiers go from 0 to (number_of_devices-1). Identifiers are managed automatically by the FlexNLPAccelerator class
    // the vector indices correspond to the accelerators' id's. This is ensured by the initialize_flexnlp_context function.
    std::vector<FlexNLPAccelerator*> accelerators_objects; // Vector containing Accelerators as objects
    int number_of_devices;

    void initialize_flexnlp_context();
  public:
    // Constructor for the case where we're using a single FlexNLP device
    FlexNLPContext();

    // Constructor for the case where we're using multiple FlexNLP devices
    FlexNLPContext(int number_of_devices);

    // Get an accelerator object by id
    FlexNLPAccelerator* get_accelerator_by_id(int device_id);
};

void FlexNLPContext::initialize_flexnlp_context(){
  for (int i = 0; i < this->number_of_devices; i++)
    this->accelerators_objects.push_back(new FlexNLPAccelerator());
}

FlexNLPContext::FlexNLPContext() : number_of_devices(1){
  std::cout << "Creating the FlexNLP Context" << '\n';
  initialize_flexnlp_context();
}

FlexNLPContext::FlexNLPContext(int number_of_devices) : number_of_devices(number_of_devices){
  std::cout << "Creating the FlexNLP Context" << '\n';
  initialize_flexnlp_context();
}

FlexNLPAccelerator* FlexNLPContext::get_accelerator_by_id(int device_id){
  FlexNLPAccelerator* accelerator = this->accelerators_objects[device_id];
  assert(accelerator->GetDeviceId() == device_id);
  return accelerator;
}
