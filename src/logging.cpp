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

#include <cctype>
#include <vector>




namespace pyoomph
{

	namespace
	{
		// --- MPI console policy -------------------------------------------------------------
		// Shared by the stream buffers below and, through the accessors in logging.hpp, by the
		// Python console wrapper in pyoomph/generic/logging.py, so that print() and oomph_info
		// follow the same rules.
		std::string g_mpi_console_mode = "off";
		int g_mpi_console_rank = 0;
		int g_mpi_console_nproc = 1;

		bool console_is_filtered() { return g_mpi_console_nproc > 1 && g_mpi_console_mode != "off"; }

		// True if this rank's ordinary output must not reach the terminal. In "condensed" mode all
		// ranks but 0 print the very same lines as rank 0 does (oomph-lib reports globally reduced
		// residuals and dof counts), so dropping them loses nothing and removes N-1 of every N lines.
		bool console_is_muted() { return console_is_filtered() && g_mpi_console_mode == "condensed" && g_mpi_console_rank != 0; }

		// oomph-lib's MPIOutputModifier writes "Processor N:   " into the stream, but only in front of
		// the *first* operand of an `oomph_info << a << b` chain: operator<< returns the raw stream, so
		// every following operand, and every embedded newline, goes out unprefixed. That is what makes
		// the multi-line solver reports fall apart under mpirun. We switch the modifier off and tag
		// whole lines ourselves instead, but strip a leftover prefix in case anything re-enables it.
		void strip_processor_prefix(std::string &line)
		{
			const std::string pre = "Processor ";
			if (line.compare(0, pre.size(), pre) != 0) return;
			size_t i = pre.size();
			while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i]))) i++;
			if (i == pre.size()) return; // "Processor" not followed by a rank number: not the modifier
			if (i >= line.size() || line[i] != ':') return;
			i++;
			while (i < line.size() && line[i] == ' ') i++;
			line.erase(0, i);
		}
	}

	//========================================================================================
	// The streambuf behind both the log-file tee and the MPI console.
	//
	// It duplicates everything written to it into the buffer it was built from (the real
	// std::cout/std::cerr) and, once set, into a log file, so pyoomph's console output can be
	// mirrored to a file without changing how the rest of the code writes to std::cout.
	//
	// When the MPI console is active it additionally collects whole lines instead of passing
	// characters straight through. That is not an optimisation: under mpirun a rank writing a
	// line in several chunks would otherwise have another rank's chunk land in the middle of it,
	// and the per-line decisions (mute this rank / tag it / indent it) need the complete line.
	// With the console off it is a pure character-for-character pass-through, so serial output is
	// unchanged.
	//========================================================================================
	class ConsoleFilterBuf : public std::streambuf
	{
	public:
		// `indent` is prepended to every non-blank line. oomph-lib's solver detail is indented
		// underneath whatever pyoomph itself printed at column 0; pyoomph's own output is not.
		// `always_tag` marks a stream whose lines carry "[rank N]" even in condensed mode: error
		// output, where several ranks write at once and an untagged line among tagged ones cannot
		// be attributed.
		ConsoleFilterBuf(std::streambuf *target, const std::string &indent, bool always_tag = false)
			: oldbuffer(target), filebuffer(NULL), indent(indent), always_tag(always_tag), last_was_blank(false) {}

		int overflow(int c) override
		{
			if (c == EOF) return c;
			const char ch = static_cast<char>(c);
			if (!console_is_filtered())
			{
				if (oldbuffer) oldbuffer->sputc(c);
				if (filebuffer) filebuffer->sputc(c);
				return c;
			}
			pending += ch;
			if (ch == '\n') emit_pending();
			// A prompt or progress marker never terminated by a newline would otherwise be held
			// back forever; flush it once it is long enough to be a line in its own right.
			else if (pending.size() > 8192) emit_pending();
			return c;
		}

		int sync() override
		{
			// Both, not either: with a log file attached the terminal would otherwise never be
			// flushed, which is what let C++ and Python output arrive out of order.
			int result = 0;
			if (oldbuffer && oldbuffer->pubsync() != 0) result = -1;
			if (filebuffer && filebuffer->pubsync() != 0) result = -1;
			return result;
		}

		// Attach (or, if filestream is NULL, detach) the log-file buffer to tee into.
		void set_file_stream(std::ostream *filestream)
		{
			filebuffer = filestream ? filestream->rdbuf() : NULL;
		}

		// Push out whatever incomplete line is still held back (end of run, or before Python
		// writes to the same terminal).
		void flush_pending()
		{
			if (!pending.empty()) emit_pending();
			sync();
		}

	private:
		void raw_write(const std::string &s, bool to_console)
		{
			if (to_console && oldbuffer) oldbuffer->sputn(s.data(), s.size());
			if (filebuffer) filebuffer->sputn(s.data(), s.size());
		}

		// Turn the buffered line into what should appear on the terminal, and write it in one go.
		void emit_pending()
		{
			std::string line;
			line.swap(pending);
			if (!line.empty() && line.back() == '\n') line.pop_back();
			strip_processor_prefix(line);

			bool blank = true;
			for (size_t i = 0; i < line.size(); i++)
				if (!std::isspace(static_cast<unsigned char>(line[i]))) { blank = false; break; }

			// always_tag streams are error streams: never muted, whatever the rank. A failure on
			// rank 3 that nobody is shown looks like a silent hang.
			const bool to_console = always_tag || !console_is_muted();
			if (blank)
			{
				// oomph-lib brackets its solver reports with blank lines, and stripping the
				// "Processor N:   " prefix turns several more into blank ones. Collapsing runs of
				// them is most of what makes the result look condensed.
				if (last_was_blank) return;
				last_was_blank = true;
				raw_write("\n", to_console);
			}
			else
			{
				last_was_blank = false;
				std::string out = indent + line;
				if (g_mpi_console_mode == "all" || always_tag)
					out = "[rank " + std::to_string(g_mpi_console_rank) + "] " + out;
				out += "\n";
				raw_write(out, to_console);
			}
			if (to_console) sync();
		}

		std::streambuf *oldbuffer;
		std::streambuf *filebuffer;
		std::string indent;
		bool always_tag;
		std::string pending;  // characters of the line currently being assembled
		bool last_was_blank;  // for collapsing runs of blank lines
	};


	// An ostream writing through a ConsoleFilterBuf, used for oomph-lib's global output streams.
	class TeeToLogFile : public std::ostream
	{
		ConsoleFilterBuf buffer;

	public:
		TeeToLogFile(std::ostream &oldstream, const std::string &indent, bool always_tag = false)
			: std::ostream(NULL), buffer(oldstream.rdbuf(), indent, always_tag)
		{
			std::ostream::rdbuf(&buffer);
		}
		void set_file_stream(std::ostream *filestream) { buffer.set_file_stream(filestream); }
		void flush_pending() { buffer.flush_pending(); }
		~TeeToLogFile() override
		{
			this->buffer.flush_pending();
			this->flush();
		}
	};


	std::ostream * g_current_log_stream=NULL; // The log file stream currently being teed into, or NULL if logging to a file is disabled

	// Tee-wrappers around the real std::cout/std::cerr; oomph-lib's global output streams are
	// redirected to point at these (see set_logging_stream) so that everything printed through
	// oomph-lib gets logged and, under mpirun, filtered. Indented by two spaces: everything
	// arriving here is solver detail belonging under a heading pyoomph printed itself.
	TeeToLogFile logged_cout(std::cout, "  ");
	TeeToLogFile logged_cerr(std::cerr, "  ", true); // errors: always attributed to a rank

	namespace
	{
		// std::cout's own buffer, replaced by this one while the MPI console is active. Going
		// through oomph_info is a convention several hundred places in src/ do not follow (and
		// need not: the messages are pyoomph's, not oomph-lib's), and every one of them would
		// otherwise print once per rank and interleave. Deliberately never deleted - it stays
		// installed in std::cout, which is used during static destruction.
		ConsoleFilterBuf *g_cout_filter = NULL;
		std::streambuf *g_original_cout_buf = NULL;
	}


	// Point oomph-lib's output streams at our tees, if they are not there already. Both the log
	// file and the MPI console need this, and either may be switched on first.
	static void ensure_streams_hooked()
	{
		if (oomph::oomph_info.stream_pt() != &logged_cout)
		{
			oomph::oomph_info.stream_pt() = &logged_cout;
			oomph::OomphLibError::set_stream_pt(&logged_cerr);
		}
	}

	// Enable (or update) mirroring of console output into `logstream` (pass NULL to stop
	// logging to a file, while still teeing through logged_cout/logged_cerr). On first
	// call, also redirects oomph-lib's internal output streams (oomph_info, OomphLibError)
	// to logged_cout/logged_cerr so that oomph-lib's own messages get logged too.
	void set_logging_stream(std::ostream * logstream)
	{
		ensure_streams_hooked();
		logged_cout.set_file_stream(logstream);
		logged_cerr.set_file_stream(logstream);
		if (g_cout_filter) g_cout_filter->set_file_stream(logstream);
		g_current_log_stream=logstream;
	}

	// Write `message` directly to the current log file (if any), bypassing the
	// tee-streams; used for messages that should only go to the log, not the console.
	void write_to_log_file(const std::string & message)
	{
		if (g_current_log_stream) *g_current_log_stream << message  << std::flush;
	}

	std::ostream * get_logging_stream() {return g_current_log_stream;}

	std::ostream * get_console_stream() { return &logged_cout; }

	void setup_mpi_console(int rank, int nproc, const std::string & mode)
	{
		g_mpi_console_rank = rank;
		g_mpi_console_nproc = nproc;
		g_mpi_console_mode = mode;
		ensure_streams_hooked();
		if (console_is_filtered() && !g_cout_filter)
		{
			g_original_cout_buf = std::cout.rdbuf();
			g_cout_filter = new ConsoleFilterBuf(g_original_cout_buf, ""); // no indent: pyoomph's own messages
			g_cout_filter->set_file_stream(g_current_log_stream);
			std::cout.rdbuf(g_cout_filter);
		}
#ifdef OOMPH_HAS_MPI
		// Take the per-line tagging away from oomph-lib and do it ourselves: its modifier only
		// prefixes the first operand of each << chain (see strip_processor_prefix above), so with
		// it left on, half the lines of a multi-line report stay untagged and interleave.
		if (console_is_filtered())
			oomph::oomph_info.output_modifier_pt() = &oomph::default_output_modifier;
#endif
	}

	std::string get_mpi_console_mode() { return g_mpi_console_mode; }
	int get_mpi_console_rank() { return g_mpi_console_rank; }
	int get_mpi_console_nproc() { return g_mpi_console_nproc; }

	void flush_console()
	{
		logged_cout.flush_pending();
		logged_cerr.flush_pending();
		if (g_cout_filter) g_cout_filter->flush_pending();
	}
}
