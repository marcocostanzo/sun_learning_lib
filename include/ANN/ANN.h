/*
    ANN Class

    Copyright 2017-2018 Università della Campania Luigi Vanvitelli

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ANN_LIB
#define ANN_LIB

//#define HYBRID_NORM

#include "ANN_Layer.h"
#include <vector>
#include <string.h>

#define NORM_TYPE_NULL 0
#define NORM_TYPE_MAPMINMAX 1
#define NORM_TYPE_MAX 2
#ifdef HYBRID_NORM 
	#define NORM_TYPE_HYBRID 3
#endif

class ANN{

	typedef TooN::Vector<> (ANN::*FF_NORM)(TooN::Vector<>);
	//int (TMyClass::*pt2ConstMember)(float, char, char) const = NULL;

private:
	unsigned int dimInput;
	unsigned int dimOutput;

	std::vector<ANN_Layer> layers;

/*======FOR MAPMINMAX========*/
	TooN::Vector<> * minInput;
	TooN::Vector<> * maxInput;
	TooN::Vector<> * minOutput;
	TooN::Vector<> * maxOutput;
/*============================*/
/*======FOR MAPMINMAX========*/
#ifdef HYBRID_NORM 
	TooN::Vector<> * inputNormMax;
	TooN::Vector<> * outputNormMax;
#endif
/*============================*/

	int normType;
	FF_NORM normFun;
	FF_NORM invNormFun;

/*==========NORMS======================*/
	TooN::Vector<> nullNorm( TooN::Vector<> in );

	TooN::Vector<> mapMinMax(TooN::Vector<> in);
	TooN::Vector<> InvMapMinMax(TooN::Vector<> in);

	TooN::Vector<> normMax(TooN::Vector<> in);
	TooN::Vector<> InvNormMax(TooN::Vector<> in);

#ifdef HYBRID_NORM 
	TooN::Vector<> normHybrid(TooN::Vector<> in);
	TooN::Vector<> invNormHybrid(TooN::Vector<> in);
#endif
/*==============================================*/

public:

/*===============CONSTRUCTORS===================*/
	ANN( std::vector<ANN_Layer> layers, TooN::Vector<> minInput, TooN::Vector<> maxInput, TooN::Vector<> minOutput, TooN::Vector<> maxOutput );

	ANN( unsigned int dimInput , unsigned int dimOutput );

	ANN(unsigned int dimInput,unsigned  int dimOutput, int normType);

	ANN( const ANN & obj ); 

	ANN( std::string config_folder);

	~ANN();
/*==============================================*/

/*=============GETTER===========================*/
	unsigned int getDimInput();
	unsigned int getDimOutput();
	unsigned int getNumLayers();

	int getNormType();

	std::vector<ANN_Layer> getLayers();

	ANN_Layer getLayer( unsigned int index );

	TooN::Vector<> getMinInput();
	TooN::Vector<> getMaxInput();
	TooN::Vector<> getMinOutput();
	TooN::Vector<> getMaxOutput();

	void display();

/*==============================================*/

#ifdef HYBRID_NORM
	TooN::Vector<> getInputNormMax();
	TooN::Vector<> getOutputNormMax();

	void setInputNormMax( TooN::Vector<> inputNormMax);
	void setOutputNormMax( TooN::Vector<> outputNormMax);

	void setInputNormMax(const char* path);
	void setOutputNormMax(const char* path);
#endif

/*=============SETTER===========================*/
	void setLayers(std::vector<ANN_Layer> layers);

	void setNormType( int nt );

	void push_back_Layer( ANN_Layer layer );
	void pop_back_Layer();
	void changeLayer(unsigned int index , ANN_Layer layer );

	void setMinInput(TooN::Vector<> minInput);
	void setMaxInput(TooN::Vector<> maxInput);
	void setMinOutput(TooN::Vector<> minOutput);
	void setMaxOutput(TooN::Vector<> maxOutput);
/*==============================================*/

/*=============RUNNER===========================*/
	TooN::Vector<> compute( TooN::Vector<> input );
/*==============================================*/

/*=============SETTER FROM FILE===========================*/
	void push_back_Layer(unsigned int n_neurons, unsigned int dimInput, const char * path, FF_OUT fun);
	void changeLayer(unsigned int index , unsigned int n_neurons, unsigned int dimInput, const char * path, FF_OUT fun);

	void setMinInput(const char* path);
	void setMaxInput(const char* path);
	void setMinMaxInput(const char* path);

	void setMinOutput(const char* path);
	void setMaxOutput(const char* path);
	void setMinMaxOutput(const char* path);
/*========================================================*/

};

ANN ANN2( unsigned int dimInput, unsigned int HL_NumNeurons, unsigned int OL_NumNeurons, TooN::Matrix<> WH, TooN::Vector<> bh, TooN::Matrix<> WO, TooN::Vector<> bo, TooN::Vector<> minInput, TooN::Vector<> maxInput,  TooN::Vector<> minOutput, TooN::Vector<> maxOutput);

ANN ANN2( unsigned int dimInput, unsigned int HL_NumNeurons, unsigned int OL_NumNeurons, const char* WH, const char* bh, const char* WO, const char* bo, const char* minInput, const char* maxInput, const char* minOutput, const char* maxOutput);

ANN ANN2( unsigned int dimInput, unsigned int HL_NumNeurons, unsigned int OL_NumNeurons, const char* WH, const char* bh, const char* WO, const char* bo, const char* minMaxInput, const char* minMaxOutput);

ANN ANN2( unsigned int dimInput, unsigned int HL_NumNeurons, unsigned int OL_NumNeurons, int normType);

ANN ANN2MAX( unsigned int dimInput, unsigned int HL_NumNeurons, unsigned int OL_NumNeurons, const char* WH, const char* bh, const char* WO, const char* bo, const char* MaxInput, const char* MaxOutput);

ANN ANN2NULLNORM( unsigned int dimInput, unsigned int HL_NumNeurons, unsigned int OL_NumNeurons, const char* WH, const char* bh, const char* WO, const char* bo);


#endif 
