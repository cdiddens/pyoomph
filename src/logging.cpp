/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt

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

The main author may be contacted at c.diddens@utwente.nl

================================================================================*/


#include "logging.hpp"
#include "oomph_utilities.h"




namespace pyoomph
{
  
  class TeeToLogFile: public std::ostream
  {
    struct TeeBuffer: public std::streambuf
    {        
        virtual int overflow(int c)
        {
            if (oldbuffer) oldbuffer->sputc(c);
            if (filebuffer) filebuffer->sputc(c);                          
            return c;
        }

        int sync() {             
          if (filebuffer) return this->filebuffer->pubsync();
          return this->oldbuffer->pubsync(); 
        }

        TeeBuffer(std::ostream & oldstream)
        {
            oldbuffer = oldstream.rdbuf();
            filebuffer = NULL;
        }

        void set_file_stream(std::ostream * filestream)
        {
            if (filestream) filebuffer = filestream->rdbuf();
            else filebuffer = NULL;
            
        }

        private:
            std::streambuf* oldbuffer;
            std::streambuf* filebuffer;
            
    };  
    TeeBuffer buffer;
    public: 
        TeeToLogFile(std::ostream & oldstream ) :  std::ostream(NULL), buffer(oldstream)
        {
            std::ostream::rdbuf(&buffer);
        }   
        void set_file_stream(std::ostream * filestream)
        {
            buffer.set_file_stream(filestream);
        }
        ~TeeToLogFile()
        {
            this->buffer.pubsync();
            this->flush();
        }

        
};


  std::ostream * g_current_log_stream=NULL;

  TeeToLogFile logged_cout(std::cout);
  TeeToLogFile logged_cerr(std::cerr);


  void set_logging_stream(std::ostream * logstream) 
  {    
    
    if (oomph::oomph_info.stream_pt()!=&logged_cout) 
    {
      oomph::oomph_info.stream_pt()=&logged_cout;
      oomph::OomphLibError::set_stream_pt(&logged_cerr);
    }    
    logged_cout.set_file_stream(logstream);
    logged_cerr.set_file_stream(logstream);
    g_current_log_stream=logstream;
  }

  void write_to_log_file(const std::string & message)
  {
    if (g_current_log_stream) *g_current_log_stream << message  << std::flush;  
  }

  std::ostream * get_logging_stream() {return g_current_log_stream;}
}
