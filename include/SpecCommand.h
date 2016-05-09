#ifndef _H_SPEC_CMD_
#define _H_SPEC_CMD_

enum SpecCommandType {ScheduleSpec, DataStorageSpec};
enum Command {
	SCHED_CMD_TILE, 	// Tile loop dimensions
	SCHED_CMD_PAR,  	// Parallelize a loop dimension
	SCHED_CMD_VEC,		// Vectorize a loop dimension
	SCHED_CMD_GPU,		// Map a loop dimension to GPU
	SCHED_CMD_DIST,		// Distribute a loop dimension
	SCHED_CMD_REORDER, 	// Reorder loop dimensions
	SCHED_CMD_SPLIT_DIMS,	// Split a loop dimension into two dimensions
	SCHED_CMD_FUSE_DIMS,	// Fuse two loop dimensions into one
	SCHED_CMD_UNROLL,	// Unroll a loop dimension by a degree
	SCHED_CMD_TIME,		// A map that provides the schedule of a statement
	STORE_CMD_ALLOC_IN,	// Multidimensional location
	STORE_CMD_ALLOC_AT,	// Mutidumensional time
	STORE_CMD_SIZE,		// Indicates array size
	STORE_CMD_STORE_IN,	// A map indicating where to store memory elements

	};


class SpecCommand {
public:
	
private:
	SpecCommandType type;
	Command cmd;

}

#endif
