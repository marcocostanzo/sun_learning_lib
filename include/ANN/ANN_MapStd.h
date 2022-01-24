/*
    ANN MAP STD LAYER

    Copyright 2019 Università della Campania Luigi Vanvitelli

	Author: Marco Costanzo <marco.costanzo@unicampania.it>

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

#ifndef ANN_MAPSTD_LAYER_H
#define ANN_MAPSTD_LAYER_H

#include <ANN/ANN_Layer_Interface.h>

class ANN_MapStd : public ANN_Layer_Interface
{

private:

ANN_MapStd(); //No default constructor

protected:

TooN::Vector<> mean_;
TooN::Vector<> std_dev_;
TooN::Vector<> output_;
bool b_apply;

public:

ANN_MapStd( 
    const TooN::Vector<>& mean, 
    const TooN::Vector<>& std_dev,
    bool b_reverse = false
)
:std_dev_(std_dev),
mean_(mean),
output_(TooN::Zeros(std_dev.size())),
b_apply(!b_reverse)
{
    if(!check_sizes())
    {
        std::cout << "ERROR IN SIZE OF ANN_MapStd" << std::endl;
        exit(-1);
    }
}

ANN_MapStd( const ANN_MapStd& l) = default;

virtual ANN_MapStd* clone() const override
{
    return new ANN_MapStd(*this);
}

virtual ~ANN_MapStd() override = default;

/*=============RUNNER===========================*/

inline virtual const TooN::Vector<>& compute( const TooN::Vector<>& input ) override
{
    if( b_apply ){
        return apply( input );
    } else {
        return reverse( input );
    }
}

/*
inline virtual TooN::Vector<> compute( const TooN::Vector<>& input ) const override
{
    if( b_apply ){
        return apply( input );
    } else {
        return reverse( input );
    }
}
*/

/*==============================================*/

/*=============METHODS===========================*/

inline virtual const TooN::Vector<>& apply( const TooN::Vector<>& input )
{
    for (int s = 0; s < mean_.size(); s++)
    {
		output_[s] = ((input[s] - mean_[s]) / std_dev_[s]); 
	}
	return output_;
}

/*
inline virtual TooN::Vector<> apply( const TooN::Vector<>& input ) const
{
    TooN::Vector<> output(TooN::Zeros(mean_.size()));
    for (int s = 0; s < mean_.size(); s++)
    {
		output[s] = ((input[s] - mean_[s]) * (1.0 / std_dev_[s])); 
	}
	return output;
}
*/

inline virtual const TooN::Vector<>& reverse( const TooN::Vector<>& input )
{
    for (int h = 0; h < mean_.size(); h++)
    {  
	    output_[h] = (input[h] * std_dev_[h]) + mean_[h];
	}
	return output_;
}

/*
inline virtual TooN::Vector<> reverse( const TooN::Vector<>& input ) const
{
    TooN::Vector<> output(TooN::Zeros(mean_.size()));
    for (int h = 0; h < mean_.size(); h++)
    {  
	    output[h] = (input_[h] * std_dev_[h]) + mean_[h];
	}
	return output;
}
*/

virtual bool check_sizes() const
{
    if(mean_.size() != std_dev_.size()){
        return false;
    }
    if(output_.size() != std_dev_.size()){
        return false;
    }
    return true;
}

/*==============================================*/

};

using ANN_MapStd_Ptr = std::unique_ptr<ANN_MapStd>;

#endif