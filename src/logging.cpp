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

	// An ostream that duplicates everything written to it into two underlying streambufs: the
	// stream it was built from (e.g. the real std::cout/std::cerr, via `oldbuffer`) and,
	// once set, an optional log file (`filebuffer`). Used to transparently mirror pyoomph's
	// console output into a log file without changing how the rest of the code writes to
	// std::cout/std::cerr (see logged_cout/logged_cerr and set_logging_stream below, which
	// redirect oomph-lib's output streams through instances of this class).
	class TeeToLogFile : public std::ostream
	{
		// The streambuf that actually performs the "tee": every character written is
		// forwarded to the original buffer and, if present, to the log-file buffer.
		struct TeeBuffer : public std::streambuf
		{
			int overflow(int c) override
			{
				if (oldbuffer) oldbuffer->sputc(c);
				if (filebuffer) filebuffer->sputc(c);
				return c;
			}

			int sync() override
			{
				if (filebuffer) return this->filebuffer->pubsync();
				return this->oldbuffer->pubsync();
			}

			TeeBuffer(std::ostream &oldstream)
			{
				oldbuffer = oldstream.rdbuf();
				filebuffer = NULL;
			}

			// Attach (or, if filestream is NULL, detach) the log-file buffer to tee into.
			void set_file_stream(std::ostream *filestream)
			{
				if (filestream) filebuffer = filestream->rdbuf();
				else filebuffer = NULL;
			}

		private:
			std::streambuf *oldbuffer;
			std::streambuf *filebuffer;
		};
		TeeBuffer buffer;

	public:
		// Wrap oldstream (e.g. std::cout) so that writes to *this go through `buffer`,
		// which forwards them to oldstream's original buffer.
		TeeToLogFile(std::ostream &oldstream) : std::ostream(NULL), buffer(oldstream)
		{
			std::ostream::rdbuf(&buffer);
		}
		void set_file_stream(std::ostream *filestream)
		{
			buffer.set_file_stream(filestream);
		}
		~TeeToLogFile() override
		{
			this->buffer.pubsync();
			this->flush();
		}
	};


	std::ostream * g_current_log_stream=NULL; // The log file stream currently being teed into, or NULL if logging to a file is disabled

	// Tee-wrappers around the real std::cout/std::cerr; oomph-lib's global output streams
	// are redirected to point at these (see set_logging_stream) so that everything printed
	// through oomph-lib also reaches the log file once one is set.
	TeeToLogFile logged_cout(std::cout);
	TeeToLogFile logged_cerr(std::cerr);


	// Enable (or update) mirroring of console output into `logstream` (pass NULL to stop
	// logging to a file, while still teeing through logged_cout/logged_cerr). On first
	// call, also redirects oomph-lib's internal output streams (oomph_info, OomphLibError)
	// to logged_cout/logged_cerr so that oomph-lib's own messages get logged too.
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

	// Write `message` directly to the current log file (if any), bypassing the
	// tee-streams; used for messages that should only go to the log, not the console.
	void write_to_log_file(const std::string & message)
	{
		if (g_current_log_stream) *g_current_log_stream << message  << std::flush;
	}

	std::ostream * get_logging_stream() {return g_current_log_stream;}
}
