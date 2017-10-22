using DataFrames
using Gadfly

function makebenchmarkplot()
    # Load benchmark data from file
    benchmarks = readtable("makeplots/benchmarks.csv", header=false, names=[:language, :benchmark, :time])

    # Capitalize and decorate language names from datafile
    dict = Dict("c"=>"C", "julia"=>"Julia", "lua"=>"LuaJIT", "fortran"=>"Fortran", "java"=>"Java",
                "javascript"=>"JavaScript", "matlab"=>"Matlab", "mathematica"=>"Mathematica", 
                "python"=>"Python", "octave"=>"Octave", "r"=>"R", "go"=>"Go")

    benchmarks[:language] = [dict[lang] for lang in benchmarks[:language]]

    # Normalize benchmark times by C times
    ctime = benchmarks[benchmarks[:language].== "C", :]
    benchmarks = join(benchmarks, ctime, on=:benchmark)
    delete!(benchmarks, :language_1)
    rename!(benchmarks, :time_1, :ctime)
    benchmarks[:normtime] = benchmarks[:time] ./ benchmarks[:ctime];

    # Compute the geometric mean for each language
    langs = [];
    means = [];
    priorities = [];
    for lang in values(dict)
        data = benchmarks[benchmarks[:language].== lang, :]
        gmean = geomean(data[:normtime])
        push!(langs, lang)
        push!(means, gmean)
        if (lang == "C")
            push!(priorities, 1)
        elseif (lang == "Julia")
            push!(priorities, 2)        
        else
            push!(priorities, 3)
        end
    end

    # Add the geometric means back into the benchmarks dataframe
    langmean = DataFrame(language=langs, geomean = means, priority = priorities)
    benchmarks = join(benchmarks, langmean, on=:language)

    # Put C first, Julia second, and sort the rest by geometric mean
    sort!(benchmarks, cols=[:priority, :geomean]);

    p = plot(benchmarks,
             x = :language,
             y = :normtime,
             color = :benchmark,
             Scale.y_log10,
             Guide.ylabel("execution time, C=1"),
             Guide.xlabel(nothing),
             Coord.Cartesian(xmin=1,xmax=12.5,ymin=-1,ymax=5),
             Theme(
                   guide_title_position = :left,
                   colorkey_swatch_shape = :circle,
                   minor_label_font = "Georgia",
                   major_label_font = "Georgia",
                   ),
    )
    draw(SVG(8inch,8inch/golden), p)

    means, langs
end

function listgeomean(means, langs)
    perm = sortperm(means)
    println("cputime\tlang") 
    for p in perm
        #println("$(means[p])\t$(langs[p])")
        @printf "%0.3f\t%s\n" means[p] langs[p]
        #@show means[p], langs[p]
    end
end
