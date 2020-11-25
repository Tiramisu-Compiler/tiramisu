#include <iostream>
#include <vector>

// TODO:FLEXNLP change the include according to the changes to
// the interface (if a different header file has to be used with the changes)
#include "pe_int8_top.h" // Behavioral Interface provided by Daniel
// Class for keeping track of the flexnlp's context (accelerators used)
// it contains all of the accelerator objects instantiated
// currently, the PETop class is being used, but it's better to have one abstract class
// that all of the accelerator objects heritate from (as PE is only one type)
class FlexNLPContext {
  private:
    // Accelerators' identifiers go from 0 to (number_of_devices-1).
    // Identifiers are managed automatically by the class
    // the vector indices correspond to the accelerators' id's. This is ensured by the initialize_flexnlp_context function.
    std::vector<PETop*> accelerators_objects; // Vector containing Accelerators as objects
    std::vector<bool> accelerators_availability; // Vector containing Accelerators as objects

    int number_of_devices; // Number of devices to be used in the tiramisu-flexnlp program

    // Instantiates the Accelerators objects and stores them in the accelerators_objects vector
    void initialize_flexnlp_context();
  public:
    // Constructor for the case where we're using a single FlexNLP device
    FlexNLPContext();

    // Constructor for the case where we're using multiple FlexNLP devices
    FlexNLPContext(int number_of_devices);

    // Get an accelerator object by id
    PETop* get_accelerator_by_id(int device_id);

    // Get the number of accelerators
    int get_number_of_devices();

    // Set the accelerator availability
    void set_accelerator_availability(int device_id, bool availability);

    // Instead of calling get_accelerator_by_id then set_accelerator_availability to set it to false
    // this function does the two in one call
    PETop* use_accelerator(int device_id);

    // This function sets an accelerator's availability to true
    void release_accelerator(int device_id);

    // Check availability
    bool isAvailable(int device_id);
};

void FlexNLPContext::initialize_flexnlp_context(){
  for (int i = 0; i < this->number_of_devices; i++){
    this->accelerators_objects.push_back(new PETop(i));
    this->accelerators_availability.push_back(true);
  }
}

FlexNLPContext::FlexNLPContext() : number_of_devices(1){
  std::cout << "Creating the FlexNLP Context" << '\n';
  initialize_flexnlp_context();
}

FlexNLPContext::FlexNLPContext(int number_of_devices) : number_of_devices(number_of_devices){
  std::cout << "Creating the FlexNLP Context" << '\n';
  initialize_flexnlp_context();
}

PETop* FlexNLPContext::get_accelerator_by_id(int device_id){
  PETop* accelerator = this->accelerators_objects[device_id];
  assert(accelerator->GetDeviceId() == device_id);
  return accelerator;
}

int FlexNLPContext::get_number_of_devices(){
  return this->number_of_devices;
}

void FlexNLPContext::set_accelerator_availability(int device_id, bool availability){
  this->accelerators_availability[device_id] = availability;
}

PETop* FlexNLPContext::use_accelerator(int device_id){
  this->accelerators_availability[device_id] = false;
  PETop* accelerator = this->accelerators_objects[device_id];
  assert(accelerator->GetDeviceId() == device_id);
  return accelerator;
}

void FlexNLPContext::release_accelerator(int device_id){
  this->accelerators_availability[device_id] = true;
}

bool FlexNLPContext::isAvailable(int device_id){
  return this->accelerators_availability[device_id];
}
